#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>


using namespace std;
using namespace cv;
cv::Mat global_dbg_img;

void  generateEllipseCandidates(
    cv::Mat& I, // input grayscale 8 bit image
    int edge_process_select, // edge detector: 1 - sobel, 2 - canny  
    int specified_polarity, // 1 - ellipse lighter than background, -1 - ellipse darker than background
    cv::Mat& candidates_out, // ellipse candidates n x 5 
    cv::Mat& edgeimg_out, // edges image
    cv::Mat& gradient_vec_out, // gradients for each edge point n x 2
    cv::Mat& ls_mat); //debug image 

// ----------------------------------------------------
// Point-ellipse edge distance.
// Used for edge points filtering.
// ----------------------------------------------------
double dRosin_square(cv::Mat& param, cv::Vec2d& point)
{
    double dmin;
    double ae2 = param.at<double>(2) * param.at<double>(2);
    double be2 = param.at<double>(3) * param.at<double>(3);
    double x = point[0] - param.at<double>(0);
    double y = point[1] - param.at<double>(1);
    double xp = x * cos(-param.at<double>(4)) - y * sin(-param.at<double>(4));
    double yp = x * sin(-param.at<double>(4)) + y * cos(-param.at<double>(4));
    double fe2 = ae2 - be2;
    double X = xp * xp;
    double Y = yp * yp;
    double delta = (X + Y + fe2) * (X + Y + fe2) - 4 * fe2 * X;
    double A = (X + Y + fe2 - sqrt(delta)) / 2;
    double ah = sqrt(A);
    double bh2 = fe2 - A;
    double term = A * be2 + ae2 * bh2;
    double xi = ah * sqrt(ae2 * (be2 + bh2) / term);
    double yi = param.at<double>(3) * sqrt(bh2 * (ae2 - A) / term);
    std::vector<double> d(4, 0);
    d[0] = (xp - xi) * (xp - xi) + (yp - yi) * (yp - yi);
    d[1] = (xp - xi) * (xp - xi) + (yp + yi) * (yp + yi);
    d[2] = (xp + xi) * (xp + xi) + (yp - yi) * (yp - yi);
    d[3] = (xp + xi) * (xp + xi) + (yp + yi) * (yp + yi);
    dmin = *std::min_element(d.begin(), d.end());
    return dmin;
}

// ----------------------------------------------------
// compute the points' normals belong to an ellipse,
// the normals have been already normalized. 
// param: [x0 y0 a b phi] .
// points : [xi yi] , n x 2
// ----------------------------------------------------
void computePointAngle(cv::Mat& ellipse,
    std::vector<cv::Vec2d>& points,
    cv::Mat& ellipse_normals)
{
    // convert[x0 y0 a b phi] to Ax ^ 2 + Bxy + Cy ^ 2 + Dx + Ey + F = 0
    double a_square = ellipse.at<double>(2) * ellipse.at<double>(2);
    double b_square = ellipse.at<double>(3) * ellipse.at<double>(3);
    double sin_phi = sin(ellipse.at<double>(4));
    double cos_phi = cos(ellipse.at<double>(4));
    double sin_square = sin_phi * sin_phi;
    double cos_square = cos_phi * cos_phi;
    double A = b_square * cos_square + a_square * sin_square;
    double B = (b_square - a_square) * sin_phi * cos_phi * 2;
    double C = b_square * sin_square + a_square * cos_square;
    double D = -2 * A * ellipse.at<double>(0) - B * ellipse.at<double>(1);
    double E = -2 * C * ellipse.at<double>(1) - B * ellipse.at<double>(0);

    // calculate points' normals to ellipse
    ellipse_normals = cv::Mat(points.size(), 2, CV_64FC1);
    for (int i = 0; i < points.size(); ++i)
    {
        double y = C * points[i][1] + B / 2 * points[i][0] + E / 2;
        double x = A * points[i][0] + B / 2 * points[i][1] + D / 2;
        double angles = atan2(y, x);
        ellipse_normals.at<double>(i, 0) = cos(angles);
        ellipse_normals.at<double>(i, 1) = sin(angles);
    }
}

// ----------------------------------------------------
// Matrix-Matrix dot product
// ----------------------------------------------------
void dot(cv::Mat& A, cv::Mat& B, Mat& Res)
{
    Res = cv::Mat(A.rows, 1, CV_64FC1);
    for (int i = 0; i < A.rows; ++i)
    {
        double sum = 0;
        for (int j = 0; j < A.cols; ++j)
        {
            sum += A.at<double>(i, j) * B.at<double>(i, j);
        }
        Res.at<double>(i) = sum;
    }
}

// ----------------------------------------------------
// Histogram computation
// ----------------------------------------------------
void histc(std::vector<size_t>& tt, size_t tbins, std::vector<size_t>& h)
{
    h = std::vector<size_t>(tbins, 0);
    for (auto t : tt)
    {
        h[t]++;
    }
}

// ----------------------------------------------------
// compute continous arcs pieces
// ----------------------------------------------------
void takeInliers(std::vector<cv::Vec2d>& x, cv::Vec2d& center, float tbins, std::vector<size_t>& idx)
{
    std::vector<double> theta(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        theta[i] = atan2(x[i][1] - center[1], x[i][0] - center[0]);
    }
    double tmin = -CV_PI;
    double tmax = CV_PI;
    std::vector<size_t> bin_number(x.size());
    for (int i = 0; i < theta.size(); ++i)
    {
        // bin number for x[i] normalized to range [0:tbins-1]
        bin_number[i] = round((theta[i] - tmin) / (tmax - tmin) * (tbins - 1));
    }
    // compute histogram of normal angles distribution
    std::vector<size_t> histogram;
    histc(bin_number, tbins, histogram);
    // label of connected histogram bins group
    std::vector<size_t> component_label(tbins, 0);
    // number of bins in each component
    std::vector<size_t> compSize(tbins, 0);
    // number of components
    int nComps = 0;
    // compute components
    std::vector<size_t>  queue(tbins, 0);
    const double du[2] = { -1, 1 };
    size_t front = 0;
    size_t rear = 0;
    for (int i = 0; i < tbins; ++i)
    {
        if (histogram[i] > 0 && component_label[i] == 0)
        {
            nComps = nComps + 1;
            component_label[i] = nComps;
            front = 0;
            rear = 0;
            queue[front] = i;

            while (front <= rear)
            {
                size_t u = queue[front];
                front++;
                for (int j = 0; j < 2; ++j)
                {
                    int v = u + du[j];
                    if (v == -1)
                    {
                        v = tbins - 1;
                    }
                    if (v >= tbins)
                    {
                        v = 0;
                    }
                    if (component_label[v] == 0 && histogram[v] > 0)
                    {
                        rear++;
                        queue[rear] = v;
                        component_label[v] = nComps;
                    }
                }
            }

            std::set<size_t> idx_list;
            for (size_t j = 0; j < component_label.size(); ++j)
            {
                if (component_label[j] == nComps)
                {
                    idx_list.insert(j);
                }
            }
            size_t sum = 0;
            for (size_t j = 0; j < bin_number.size(); ++j)
            {
                if (idx_list.find(bin_number[j]) != idx_list.end())
                {
                    sum++;
                }
            }
            compSize[nComps] = sum;
        }
    }

    size_t maxCompSize = 0;
    // as components indexing start from 1 we need to add 1 to size
    compSize.resize(nComps + 1);
    if (nComps > 0)
    {
        maxCompSize = *std::max_element(compSize.begin(), compSize.end());
        if (maxCompSize > 10)
        {
            std::vector<size_t> validComps;
            for (int j = 0; j < compSize.size(); ++j)
            {
                if (compSize[j] > 0)
                {
                    validComps.push_back(j);
                }
            }
            std::vector<size_t> validBins;
            for (int j = 0; j < component_label.size(); ++j)
            {
                for (int k = 0; k < validComps.size(); ++k)
                {
                    if (component_label[j] == validComps[k])
                    {
                        validBins.push_back(j);
                    }
                }
            }

            idx.clear();
            for (int k = 0; k < validBins.size(); ++k)
            {
            for (int j = 0; j < bin_number.size(); ++j)
                {
                    if (bin_number[j] == validBins[k])
                    {
                        idx.push_back(j);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------
// 
// ----------------------------------------------------
float calcuCompleteness(std::vector<cv::Vec2d>& x, cv::Vec2d& center, size_t tbins)
{
    float completeness = 0;
    std::vector<double> theta(x.size(), 0);
    for (int i = 0; i < x.size(); ++i)
    {
        theta[i] = atan2(x[i][1] - center[1], x[i][0] - center[0]);
    }
    const double tmin = -CV_PI;
    const double tmax = CV_PI;
    std::vector<size_t> tt(theta.size(), 0);
    for (int i = 0; i < x.size(); ++i)
    {
        tt[i] = round((theta[i] - tmin) / (tmax - tmin) * (tbins - 1));
    }
    std::vector<size_t> h(tbins, 0);
    histc(tt, tbins, h);
    float h_greatthanzero_num = 0;
    for (auto v : h)
    {
        if (v > 0)
        {
            h_greatthanzero_num++;
        }
    }
    completeness = h_greatthanzero_num / tbins;
    return completeness;
}

struct point2d
{
    double x, y;
};
// ----------------------------------------------------
// 
// ----------------------------------------------------
//input : (xi,yi)
//output: x0,y0,a,b,phi,ellipara
//successfull, return 1; else return 0
int fitEllipse(point2d* dataxy, int datanum, double* ellipara);
// ----------------------------------------------------
// 
// ----------------------------------------------------
void fitEllipse(std::vector<cv::Vec2d>& points, cv::Mat& ellipse, int& info)
{
    info = fitEllipse((point2d*)points.data(), points.size(), (double*)ellipse.data);
}


void subEllipseDetection(cv::Mat& list, // N x 5
    std::vector<cv::Vec2d>& points,
    cv::Mat& normals,
    double distance_tolerance,
    double normal_tolerance,
    double Tmin,
    double angleCoverage,
    cv::Mat& E,
    std::vector<double>& angleLoop,
    std::vector<size_t>& mylabels,
    std::vector<size_t>& labels,
    cv::Mat& ellipses,
    std::vector<int>& validCandidates)
{
    labels = std::vector<size_t>(points.size(), 0);
    mylabels = std::vector<size_t>(points.size(), 0);
    ellipses = cv::Mat();
    int ellipse_polarity = 0;

    // max_dis = max(points) - min(points);    
    // double maxSemiMajor = max(max_dis);
    // double maxSemiMinor = min(max_dis);

    double distance_tolerance_square = distance_tolerance * distance_tolerance;
    validCandidates = std::vector<int>(list.rows, 1);
    cv::Mat convergence = list.clone();
    for (size_t i = 0; i < list.rows; ++i)
    {
        cv::Vec2d ellipseCenter(list.at<double>(i, 0), list.at<double>(i, 1));
        cv::Vec2d ellipseAxes(list.at<double>(i, 2), list.at<double>(i, 3));;
        double ellipsePhi = list.at<double>(i, 4);
        //ellipse circumference is approximate pi * (1.5*sum(ellipseAxes)-sqrt(ellipseAxes(1)*ellipseAxes(2))
        double tbins = min(180.0, floor(CV_PI * (1.5 * (ellipseAxes[0] + ellipseAxes[1]) - sqrt(ellipseAxes[0] * ellipseAxes[1])) * Tmin));
        std::vector<size_t> inliers;
        std::vector<size_t> i_dx;
        for (int j = 0; j < points.size(); ++j)
        {
            if (points[i][0] >= (ellipseCenter[0] - ellipseAxes[0] - distance_tolerance) &&
                points[i][0] <= (ellipseCenter[0] + ellipseAxes[0] + distance_tolerance) &&
                points[i][1] >= (ellipseCenter[1] - ellipseAxes[0] - distance_tolerance) &&
                points[i][1] <= (ellipseCenter[1] + ellipseAxes[0] + distance_tolerance))
            {
                i_dx.push_back(j);
            }
        }
        // ===============================================================================================================================================================       
        //i_dx = find(points(:, 1) >= (ellipseCenter(1) - ellipseAxes(1) - distance_tolerance) & points(:, 1) <= (ellipseCenter(1) + ellipseAxes(1) + distance_tolerance) & points(:, 2) >= (ellipseCenter(2) - ellipseAxes(1) - distance_tolerance) & points(:, 2) <= (ellipseCenter(2) + ellipseAxes(1) + distance_tolerance));

        // inliers = i_dx( labels(i_dx) == 0 && (dRosin_square(list(i, :), points(i_dx, :)) <= distance_tolerance_square));

        std::vector<size_t> labels_idx;
        std::vector<cv::Vec2d> points_idx;
        for (int j = 0; j < i_dx.size(); j++)
        {
            labels_idx.push_back(labels[i_dx[j]]);
            points_idx.push_back(points[i_dx[j]]);
        }

        for (int j = 0; j < i_dx.size(); ++j)
        {
            if (labels_idx[j] == 0 && (dRosin_square(list.row(i), points_idx[j]) <= distance_tolerance_square))
            {
                inliers.push_back(i_dx[j]);
            }
        }

        cv::Mat ellipse_normals;
        std::vector<cv::Vec2d> inlierPts;
        cv::Mat inlierNormals = cv::Mat::zeros(inliers.size(), 2, CV_64FC1);
        for (int i = 0; i < inliers.size(); ++i)
        {
            inlierPts.push_back(points[inliers[i]]);
            inlierNormals.at<double>(i, 0) = normals.at<double>(inliers[i], 0);
            inlierNormals.at<double>(i, 1) = normals.at<double>(inliers[i], 1);
        }
        computePointAngle(list.row(i), inlierPts, ellipse_normals);

        cv::Mat p_dot_temp;
        dot(inlierNormals, ellipse_normals, p_dot_temp);
        size_t p_cnt = 0;
        for (int j = 0; j < p_dot_temp.rows; ++j)
        {
            if (p_dot_temp.at<double>(j) > 0)
            {
                ++p_cnt;
            }
        }

        std::vector<size_t> inliers_tmp;
        if (p_cnt > inliers.size() * 0.5)
        {
            ellipse_polarity = -1;
            for (int k = 0; k < inliers.size(); ++k)
            {
                if (p_dot_temp.at<double>(k) >= 0.923879532511287)
                {
                    inliers_tmp.push_back(inliers[k]);
                }
            }
        }
        else
        {
            ellipse_polarity = 1;
            for (int k = 0; k < inliers.size(); ++k)
            {
                if (p_dot_temp.at<double>(k) <= 0.923879532511287)
                {
                    inliers_tmp.push_back(inliers[k]);
                }
            }
        }
        std::swap(inliers_tmp, inliers);
        inliers_tmp.clear();

        std::vector<size_t> inliers2 = inliers;
        std::vector<size_t> inliers3;
        //=================================================================================================================================================================        
        inlierPts.clear();
        for (int i = 0; i < inliers.size(); ++i)
        {
            inlierPts.push_back(points[inliers[i]]);
        }

        std::vector<size_t> idx;
        takeInliers(inlierPts, ellipseCenter, tbins, idx);

        for (auto id : idx)
        {
            inliers_tmp.push_back(inliers[id]);
        }

        inlierPts.clear();
        for (int i = 0; i < inliers.size(); ++i)
        {
            inlierPts.push_back(points[inliers[i]]);
        }
        cv::Mat new_ellipse(1, 5, CV_64FC1);
        int new_info = 0;

        fitEllipse(inlierPts, new_ellipse, new_info);
        /*
        if (new_info == 1)
        {
            if ((  (pow(new_ellipse.at<double>(0) - ellipseCenter[0],2) +
                    pow(new_ellipse.at<double>(1) - ellipseCenter[1],2)) <= 16 * distance_tolerance_square) &&
                ((  pow(new_ellipse.at<double>(2) - ellipseAxes[0],2) + pow(new_ellipse.at<double>(3) - ellipseAxes[1],2)) <= 16 * distance_tolerance_square) &&
                   (fabs(new_ellipse.at<double>(4) - ellipsePhi) <= 0.314159265358979))
            {
                computePointAngle(new_ellipse, points, ellipse_normals);
                i_dx = find(points(:, 1) >= (new_ellipse(1) - new_ellipse(3) - distance_tolerance) & points(:, 1) <= (new_ellipse(1) + new_ellipse(3) + distance_tolerance) & points(:, 2) >= (new_ellipse(2) - new_ellipse(3) - distance_tolerance) & points(:, 2) <= (new_ellipse(2) + new_ellipse(3) + distance_tolerance));
                ellipse_normals = computePointAngle(new_ellipse, points(i_dx, :));
                newinliers = i_dx(labels(i_dx) == 0 & (dRosin_square(new_ellipse, points(i_dx, :)) <= distance_tolerance_square & ((dot(normals(i_dx, :), ellipse_normals, 2) * (-ellipse_polarity)) >= 0.923879532511287)));
                newinliers = newinliers(takeInliers(points(newinliers, :), new_ellipse(1:2), tbins));
                if (length(newinliers) >= length(inliers))
                {
                    inliers = newinliers;
                    inliers3 = newinliers;
                    [new_new_ellipse, new_new_info] = fitEllipse(points(inliers, 1), points(inliers, 2));
                    if (new_new_info == 1)
                    {
                        new_ellipse = new_new_ellipse;
                    }
                    //=======================================================================
                }
            }
        }
        else
        {
            new_ellipse = list(i, :); // candidates
        }

        if (inliers.size() >= floor(CV_PI * (1.5 * sum(new_ellipse(3:4)) - sqrt(new_ellipse(3) * new_ellipse(4))) * Tmin))
        {
            convergence(i, :) = new_ellipse;
            if (any((sqrt(sum((convergence(1 : i - 1, 1 : 2) - repmat(new_ellipse(1:2), i - 1, 1)) . ^ 2, 2)) <= distance_tolerance)
                && (sqrt(sum((convergence(1 : i - 1, 3 : 4) - repmat(new_ellipse(3:4), i - 1, 1)) . ^ 2, 2)) <= distance_tolerance)
                && (abs(convergence(1 : i - 1, 5) - repmat(new_ellipse(5), i - 1, 1)) <= 0.314159265358979)))
            {
                validCandidates(i) = false;
            }
            completeOrNot = calcuCompleteness(points(inliers, :), new_ellipse(1:2), tbins) >= angleCoverage;
            if (new_info == 1 && new_ellipse(3) < maxSemiMajor && new_ellipse(4) < maxSemiMinor && completeOrNot)
            {
                if (all((sqrt(sum((ellipses(:, 1 : 2) - repmat(new_ellipse(1:2), size(ellipses, 1), 1)) . ^ 2, 2)) > distance_tolerance)
                    || (sqrt(sum((ellipses(:, 3 : 4) - repmat(new_ellipse(3:4), size(ellipses, 1), 1)) . ^ 2, 2)) > distance_tolerance)
                    || (abs(ellipses(:, 5) - repmat(new_ellipse(5), size(ellipses, 1), 1)) >= 0.314159265358979)))
                {
                    labels(inliers) = size(ellipses, 1) + 1;
                    //==================================================================
                    if (all(inliers3) == 1)
                    {
                        mylabels(inliers3) = size(ellipses, 1) + 1;
                    }
                    //==================================================================
                    ellipses = [ellipses; new_ellipse];
                    validCandidates[i] = false;
                }
            }
        }
        else
        {
            validCandidates[i] = false;
        }
*/
    }

}


void updateInlierPts(std::vector<cv::Vec2d>& srcPts, std::vector<size_t>& inlier_idxs, std::vector<cv::Vec2d>& inlierPts)
{
    std::vector<cv::Vec2d> tmpPts;
    tmpPts.assign(srcPts.begin(), srcPts.end());
    inlierPts.clear();
    for (auto inl : inlier_idxs)
    {
        inlierPts.push_back(tmpPts[inl]);
    }
}
// ----------------------------------------------------
// 
// ----------------------------------------------------
void ellipseDetection(cv::Mat& candidates, std::vector<cv::Vec2d>& points, cv::Mat& normals, float distance_tolerance, float normal_tolerance, float Tmin, float angleCoverage, cv::Mat& E, std::vector<size_t>& mylabels, std::vector<size_t>& labels, cv::Mat& ellipses)
{
    labels = std::vector<size_t>(points.size(), 0);
    mylabels = std::vector<size_t>(points.size(), 0);
    ellipses = cv::Mat();

    std::vector<size_t> validElls;
    std::vector<std::pair< size_t, float > > ellIndGoodness;
    vector<double> goodness;
    for (int i = 0; i < candidates.rows; ++i)
    {
        // ellipse circumference is approximate pi* (1.5 * sum(ellipseAxes) - sqrt(ellipseAxes(1) * ellipseAxes(2))
        cv::Vec2d ellipseCenter(candidates.at<double>(i, 0), candidates.at<double>(i, 1));
        cv::Vec2d ellipseAxes(candidates.at<double>(i, 2), candidates.at<double>(i, 3));
        double tbins = min(180.0, floor(CV_PI * (1.5 * (ellipseAxes[0] + ellipseAxes[1]) - sqrt(ellipseAxes[0] * ellipseAxes[1])) * Tmin));
        std::vector<size_t> s_dx;

        for (int j = 0; j < points.size(); ++j)
        {
            if (points[j][0] >= (ellipseCenter[0] - ellipseAxes[0] - 1) &&
                points[j][0] <= (ellipseCenter[0] + ellipseAxes[0] + 1) &&
                points[j][1] >= (ellipseCenter[1] - ellipseAxes[0] - 1) &&
                points[j][1] <= (ellipseCenter[1] + ellipseAxes[0] + 1))
            {
                s_dx.push_back(j);
            }
        }
        std::vector<size_t> inliers;

        for (int j = 0; j < s_dx.size(); ++j)
        {
            if (dRosin_square(candidates.row(i), points[s_dx[j]]) <= 1)
            {
                inliers.push_back(s_dx[j]);
            }
        }

        //ellipse_normals = computePointAngle(candidates(i, :), points(inliers, :));
        cv::Mat ellipse_normals;
        std::vector<cv::Vec2d> inlierPts;
        cv::Mat inliers_normals(inliers.size(), 2, CV_64FC1);
        for (int k = 0; k < inliers.size(); ++k)
        {
            inlierPts.push_back(points[inliers[k]]);
            inliers_normals.at<double>(k, 0) = normals.at<double>(inliers[k], 0);
            inliers_normals.at<double>(k, 1) = normals.at<double>(inliers[k], 1);
        }

        // ------------------------------------------------------
        // Measure angles between ellipse and edge points normals
        // ------------------------------------------------------        
        computePointAngle(candidates.row(i), inlierPts, ellipse_normals);
        // plot edge points normals
        for (int i = 0; i < inliers.size(); ++i)
        {
            cv::Point p1(inlierPts[i][0], inlierPts[i][1]);
            float nx = ellipse_normals.at<double>(i, 0) * 10;
            float ny = ellipse_normals.at<double>(i, 1) * 10;
            float nx1 = inliers_normals.at<double>(i, 0) * 10;
            float ny1 = inliers_normals.at<double>(i, 1) * 10;
            cv::Point p2(inlierPts[i][0] + nx, inlierPts[i][1] + ny);
            cv::Point p21(inlierPts[i][0] + nx1, inlierPts[i][1] + ny1);
            line(global_dbg_img, p1, p2, Scalar(255, 255, 0));
            line(global_dbg_img, p1, p21, Scalar(255, 0, 255));
        }
        cv::Mat p_dot_temp;
        dot(inliers_normals, ellipse_normals, p_dot_temp);
        size_t p_cnt = 0;
        for (int i = 0; i < p_dot_temp.rows; ++i)
        {
            if (p_dot_temp.at<double>(i) > 0)
            {
                p_cnt++;
            }
        }
        std::vector<size_t> inliers_tmp;
        if (p_cnt > inliers.size() * 0.5)
        {
            // ellipse_polarity = -1;
            for (int k = 0; k < inliers.size(); ++k)
            {
                if (p_dot_temp.at<double>(k) >= 0.923879532511287)
                {
                    inliers_tmp.push_back(inliers[k]);
                }
            }
        }
        else
        {
            // ellipse_polarity = 1;            
            for (int k = 0; k < inliers.size(); ++k)
            {
                if (p_dot_temp.at<double>(k) <= -0.923879532511287)
                {
                    inliers_tmp.push_back(inliers[k]);
                }
            }
        }
        std::swap(inliers_tmp, inliers);
        // ------------------------------------------------------
        // Filter inliers by continous sectors
        // ------------------------------------------------------
        // Update inlier points and normals
        inliers_tmp.clear();

        updateInlierPts(points, inliers, inlierPts);

        std::vector<size_t> idx;
        takeInliers(inlierPts, ellipseCenter, tbins, idx);
        std::cout << "inlierPts " << i << " : " << inlierPts.size() << std::endl;
        std::cout << "idx " << i << " : " << idx.size() << std::endl;
        for (auto id:idx)
        {
            inliers_tmp.push_back(inliers[id]);
        }
        std::swap(inliers_tmp, inliers);
        inliers_tmp.clear();
        // ------------------------------------------------------
        // compute quality of detected ellipses
        // ------------------------------------------------------
        updateInlierPts(points, inliers, inlierPts);

        float support_inliers_ratio = (float)inliers.size() / (CV_PI * (1.5 * (ellipseAxes[0] + ellipseAxes[1]) - sqrt(ellipseAxes[0] * ellipseAxes[1])));
        float completeness_ratio = calcuCompleteness(inlierPts, ellipseCenter, tbins);
        float g = sqrt(support_inliers_ratio * completeness_ratio);
        ellIndGoodness.push_back(std::make_pair(i, g));
    }
    // sort ellipses by quality
    std::sort(ellIndGoodness.begin(), ellIndGoodness.end(), [](auto& left, auto& right)
        {
            return left.second > right.second;
        });

    ellipses = cv::Mat::zeros(candidates.size(), CV_64FC1);
    for (int i = 0; i < ellIndGoodness.size(); ++i)
    {
        goodness.push_back(ellIndGoodness[i].second);
        validElls.push_back(ellIndGoodness[i].first);
        candidates.row(ellIndGoodness[i].first).copyTo(ellipses.row(i));
        std::cout << ellIndGoodness[i].first <<" : "<< ellIndGoodness[i].second << std::endl;
    }
    candidates = ellipses.clone();
    // candidates now sorted by goodness

    // plot ellipses
    cv::cvtColor(E, E, cv::COLOR_GRAY2BGR);
    for (int ei = 0; ei < candidates.rows / 3; ei++)
    {
        int x = (int)candidates.at<double>(ei, 0);
        int y = (int)candidates.at<double>(ei, 1);
        int a1 = candidates.at<double>(ei, 2);
        int a2 = candidates.at<double>(ei, 3);
        double ang = candidates.at<double>(ei, 4) * 180 / CV_PI;
        if (a1 > 0 && a2 > 0)
        {
            cv::ellipse(E, cv::Point(x, y), cv::Size(a1, a2), ang, 0, 360, (Scalar(255, 0, 0)), 2);
        }
    }
    cv::imshow("E", E);

    std::vector<double> angles_init = { 300, 210, 150, 90 };
    std::vector<double> angles;
    // angles(angles < angleCoverage) = [];
    for (auto a : angles_init)
    {
        if (a >= angleCoverage)
        {
            angles.push_back(a);
        }
    }

    if (angles.empty() || angles[angles.size() - 1] != angleCoverage)
    {
        angles.push_back(angleCoverage);
    }

    for (size_t angleLoop = 0; angleLoop < angles.size(); ++angleLoop)
    {
        std::vector<size_t> idx;
        for (int i = 0; i < labels.size(); ++i)
        {
            if (labels[i] == 0)
            {
                idx.push_back(i);
            }
        }
        if (idx.size() < 2 * CV_PI * (6 * distance_tolerance) * Tmin)
        {
            break;
        }

        std::vector<cv::Vec2d> idxPts;
        cv::Mat idx_normals = cv::Mat(idx.size(), 2, CV_64FC1);
        for (int k = 0; k < idx.size(); ++k)
        {
            idxPts.push_back(points[idx[k]]);
            idx_normals.at<double>(k, 0) = normals.at<double>(idx[k], 0);
            idx_normals.at<double>(k, 1) = normals.at<double>(idx[k], 1);
        }
        //cv::Mat E;
        std::vector<double> anglesLoop;
        std::vector<size_t> L2;
        std::vector<size_t> L;
        cv::Mat C;
        std::vector<int> validCandidates;
        subEllipseDetection(candidates, idxPts, idx_normals, distance_tolerance, normal_tolerance, Tmin, angles[angleLoop], E, anglesLoop, L2, L, C, validCandidates);
        /*
        [L2, L, C, validCandidates] = subEllipseDetection(candidates, points(idx, :), normals(idx, :), distance_tolerance, normal_tolerance, Tmin, angles(angleLoop), E, angleLoop);

        candidates = candidates(validCandidates, :);
        // size(candidates)
        // disp(angleLoop)
        if (size(C, 1) > 0)
        {
            for i = 1 : size(C, 1)
            {
                flag = false;
                for j = 1 : size(ellipses, 1)
                {
                    if (sqrt((C(i, 1) - ellipses(j, 1)) . ^ 2 + (C(i, 2) - ellipses(j, 2)) . ^ 2) <= distance_tolerance
                        && sqrt((C(i, 3) - ellipses(j, 3)) . ^ 2 + (C(i, 4) - ellipses(j, 4)) . ^ 2) <= distance_tolerance
                        && abs(C(i, 5) - ellipses(j, 5)) <= 0.314159265358979)
                    {
                        flag = true;
                        labels(idx(L == i)) = j; % Èç¹ûÖØ¸´ÁË£¬¾Í°Ñ¸Ã±êÇ©×ªÒÆµ½Ö®Ç°µÄÔ²ÉÏÃæ¡£×¢Òâ×¢Òâ£ºidx´æµÄÊÇË÷Òý£¬labelÊÇ±ê¼Ç¸Ã±ßÔµµãÓÃÔÚÁËµÚj¸öÔ²ÉÏ£¬idx¡¢labels¶¼ÊÇÒ»Î¬ÀàÏòÁ¿(n x 1)£¬labelsÓë±ßÔµµãpoints(n x 2)×ÜÊýÒ»Ñù, ¶øidxÔò²»Ò»¶¨
                        //= ================================================ =
                        mylabels(idx(L2 == i)) = j;
                        //= ================================================ =
                        break;
                    }
                }
                if (~flag)
                {
                    labels(idx(L == i)) = size(ellipses, 1) + 1;
                    //=================================================================
                    mylabels(idx(L2 == i)) = size(ellipses, 1) + 1;
                        //=================================================================
                    ellipses = [ellipses; C(i, :)];
                }
            }
        }
        */
    }

}


//function[ellipses, L, posi] = ellipseDetectionByArcSupportLSs(I, Tac, Tr, specified_polarity)
//% input:
//% I : input image
//% Tac : elliptic angular coverage(completeness degree)
//% Tni : ratio of support inliers on an ellipse
//% output :
//    % ellipses : N by 5. (center_x, center_y, a, b, phi)
//    % reference :
//    % 1¡¢von Gioi R Grompone, Jeremie Jakubowicz, Jean -
//    % Michel Morel, and Gregory Randall, ¡°Lsd : a fast line
//    % segment detector with a false detection control., ¡± IEEE
//    % transactions on pattern analysisand machine intelligence,
//    % vol. 32, no. 4, pp. 722¨C732, 2010.

void ellipseDetectionByArcSupportLSs(cv::Mat& I, float Tac, float Tr, float specified_polarity)
{
    float angleCoverage = Tac;// default 165¡ã
    float Tmin = Tr;// default 0.6
    float unit_dis_tolerance = 2; // max([2, 0.005 * min([size(I, 1), size(I, 2)])]);% ÄÚµã¾àÀëµÄÈÝÈÌ²îÐ¡ÓÚmax(2, 0.5 % *minsize)
    float normal_tolerance = CV_PI / 9;
    cv::Mat candidates;
    cv::Mat edge;
    cv::Mat normals;
    cv::Mat lsimg;

    if (I.channels() > 1)
    {
        cv::cvtColor(I, I, cv::COLOR_BGR2GRAY);
    }
    //[candidates, edge, normals, lsimg] = generateEllipseCandidates(I, 2, specified_polarity); // 1, sobel; 2, canny
    generateEllipseCandidates(I, 2, specified_polarity, candidates, edge, normals, lsimg);
    cv::imshow("edge", edge);
    cv::imshow("lsimg", lsimg);
    candidates = candidates.t();
    if (candidates.at<double>(0) == 0)
    {
        candidates = cv::Mat();
    }
    cv::Mat posi = candidates.clone();
    normals = normals.t();//norams matrix transposition

    std::vector<cv::Vec2d> edgePts;
    for (int j = 0; j < edge.cols; ++j)
    {
        for (int i = 0; i < edge.rows; ++i)
        {
            uchar v = edge.at<uchar>(i, j);
            if (v > 0)
            {
                edgePts.push_back(cv::Vec2d(j, i));
            }
        }
    }

    // ellipses = []; L = [];    
    //[mylabels, labels, ellipses] = ellipseDetection(candidates, [x, y], normals, unit_dis_tolerance, normal_tolerance, Tmin, angleCoverage, I);
    std::vector<size_t> mylabels;
    std::vector<size_t> labels;
    cv::Mat ellipses;
    cv::Mat E = I.clone();
    ellipseDetection(candidates, edgePts, normals, unit_dis_tolerance, normal_tolerance, Tmin, angleCoverage, E, mylabels, labels, ellipses);
    std::cout << "-----------------------------------------------------------" << std::endl;

    // labels
    // size(labels)
    // size(y)    
   // L = zeros(size(I, 1), size(I, 2));
   // L(sub2ind(size(L), y, x)) = mylabels;% labels
   // figure; imshow(L == 2);% LLL
   // imwrite((L == 2), 'D:\Graduate Design\»­Í¼\edge_result.jpg');
}

int main(int argc, char* argv[])
{
    Mat image = cv::imread("../../pics/27.jpg", 1);

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    global_dbg_img = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::imshow("image", image);
    double Tac = 165;
    double Tr = 0.6;
    float E = 0;
    int polarity = 0;
    ellipseDetectionByArcSupportLSs(image, Tac, Tr, polarity);
    cv::imshow("global_dbg_img", global_dbg_img);
    cv::waitKey();
}