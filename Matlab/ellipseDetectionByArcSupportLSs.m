% ================================================================================================================================
%
% ================================================================================================================================
function [ellipses, L, posi] = ellipseDetectionByArcSupportLSs(I, Tac, Tr, specified_polarity)
%input:
% I: input image
% Tac: elliptic angular coverage (completeness degree)
% Tni: ratio of support inliers on an ellipse
%output:
% ellipses: N by 5. (center_x, center_y, a, b, phi)
    %default angleCoverage is 165
    angleCoverage = Tac;
    %default Tmin is 0.6
    Tmin = Tr;
    unit_dis_tolerance = 2;
    normal_tolerance = pi/9;
    t0 = clock;
    % Convert image to grayscale
    if(size(I,3)>1)
        I = rgb2gray(I);
    end
    # produce ellipse candidates
    [candidates, edge, normals, lsimg] = generateEllipseCandidates(I, 2, specified_polarity);%1,sobel; 2,canny
    t1 = clock;
    disp(['The time of generating ellipse candidates:',num2str(etime(t1,t0))]);
    candidates = candidates';%ellipse candidates matrix Transposition
    if(candidates(1) == 0)
        candidates =  zeros(0, 5);
    end
    % ellipses
    posi = candidates;
    % edge normal vectors (normalized))
    normals    = normals';
    % get a list of edge pixels coordinates
    [y, x]=find(edge);
    % data from cpp implementation, so move origin to (0,0) from (1,1)
    y=y-1;
    x=x-1;
    % detect ellipses (process ellipse candidates)
    [mylabels,labels, ellipses] = ellipseDetection(candidates ,[x, y], normals, unit_dis_tolerance, normal_tolerance, Tmin, angleCoverage, I);
    disp('-----------------------------------------------------------');
    disp(['running time:',num2str(etime(clock,t0)),'s']);
    warning('on', 'all');
    % labels image values are indexes of ellipse which current pixel belons to.
    L = zeros(size(I, 1), size(I, 2));
    L(sub2ind(size(L), y+1, x+1)) = mylabels;
end
% ================================================================================================================================
%
% ================================================================================================================================
function [mylabels,labels, ellipses] = ellipseDetection(candidates, points, normals, distance_tolerance, normal_tolerance, Tmin, angleCoverage, E)
    % indices of ellipsses which detected edge pixels belongs to.
    labels = zeros(size(points, 1), 1);
    mylabels = zeros(size(points, 1), 1);
    % list of detected ellipses
    ellipses = zeros(0, 5);
    % the quality maesure of detected ellipses
    goodness = zeros(size(candidates, 1), 1);
    % iterate ellipsse candidates
    for i = 1 : size(candidates,1)
        ellipseCenter = candidates(i, 1 : 2);
        ellipseAxes   = candidates(i, 3:4);
        % number of histogram bins
        % ellipse circumference is approximate pi * (1.5*sum(ellipseAxes)-sqrt(ellipseAxes(1)*ellipseAxes(2))
        % default Tmin = 0.6
        % tbins=min(180,Tmin*circumference). circumference is a number of pixels on ellipse edge.
        tbins = min( [ 180, floor( pi * (1.5*sum(ellipseAxes)-sqrt(ellipseAxes(1)*ellipseAxes(2)) ) * Tmin ) ] );
        % get edgee points inside square with side equal to long axis of ellipse
        s_dx = find( points(:,1) >= (ellipseCenter(1)-ellipseAxes(1)-1) & ...
                     points(:,1) <= (ellipseCenter(1)+ellipseAxes(1)+1) & ...
                     points(:,2) >= (ellipseCenter(2)-ellipseAxes(1)-1) & ...
                     points(:,2) <= (ellipseCenter(2)+ellipseAxes(1)+1));
        % dRosin_square computes squared point to ellipse distance
        inliers = s_dx(dRosin_square(candidates(i,:),points(s_dx,:)) <= 1);
        % compute edge pixesls normals for current ellipse inlier edge pixels
        ellipse_normals = computePointAngle(candidates(i,:),points(inliers,:));
        % angle between vectors = acos(v1.dot(v2))
        % compute angles between ellipse candidate edge normals and edge pixel inliers normals.
        p_dot_temp = dot(normals(inliers,:), ellipse_normals, 2);
        % If normals have the same directions
        p_cnt = sum(p_dot_temp>0);
        if(p_cnt > size(inliers,1)*0.5)
            %cos(pi/8) = 0.923879532511287
            inliers = inliers(p_dot_temp >= 0.923879532511287 );
        % If normals have the opposite directions
        else
            inliers = inliers(p_dot_temp <= -0.923879532511287 );
        end
        % take inlier edgee pixels, making continous arcs
        % it will give us quality estimation measure - longer continous arcs -> better ellipse
        inliers = inliers(takeInliers(points(inliers, :), ellipseCenter, tbins));
        support_inliers_ratio = length(inliers)/floor( pi * (1.5*sum(ellipseAxes)-sqrt(ellipseAxes(1)*ellipseAxes(2)) ));
        % the second quality measure is how large angle sector have desired pixels density
        completeness_ratio = calcuCompleteness(points(inliers,:),ellipseCenter,tbins)/360;
        %goodness = sqrt(r_i * r_c)
        goodness(i) = sqrt(support_inliers_ratio*completeness_ratio);
    end
    %drawEllipses(ellipses',E);ellipses
    %here we can use pseudo order to speed up
    [goodness_descending, goodness_index] = sort(goodness,1,'descend');
    candidates = candidates(goodness_index(goodness_descending>0),:);

    %default angleCoverage is 165
    angles = [300; 210; 150; 90];
    angles(angles < angleCoverage) = [];
    if (isempty(angles) || angles(end) ~= angleCoverage)
        angles = [angles; angleCoverage];
    end
    % default value of unit_dis_tolerance is 2;
    % default Tmin = 0.6
    for angleLoop = 1 : length(angles)
        % labels = 0 for free edge pixels
        idx = find(labels == 0);
        if (length(idx) < 2 * pi * (6 * distance_tolerance) * Tmin)
            break;
        end
        [L2, L, C, validCandidates] = subEllipseDetection( candidates, points(idx, :), normals(idx, :), distance_tolerance, normal_tolerance, Tmin, angles(angleLoop), E, angleLoop);
        candidates = candidates(validCandidates, :);
        % if found some ellipses
        % combine close partial elipses, detected by subEllipseDetection
        if (size(C, 1) > 0)
            % iterate them
            for i = 1 : size(C, 1)
                flag = false;
                for j = 1 : size(ellipses, 1)
                    % check if C(i) and ellipses(j) are the same
                    if (sqrt((C(i, 1) - ellipses(j, 1)) .^ 2 + (C(i, 2) - ellipses(j, 2)) .^ 2) <= distance_tolerance ...
                        && sqrt((C(i, 3) - ellipses(j, 3)) .^ 2 + (C(i, 4) - ellipses(j, 4)) .^ 2) <= distance_tolerance ...
                        && abs(C(i, 5) - ellipses(j, 5)) <= 0.314159265358979)
                        flag = true;
                        labels(idx(L == i)) = j;
                        %==================================================
                        mylabels(idx(L2 == i)) = j;
                        %==================================================
                        break;
                    end
                end
                % if it have no clones append it to list
                if (~flag)
                    labels(idx(L == i)) = size(ellipses, 1) + 1;
                    %=================================================================
                    mylabels(idx(L2 == i)) = size(ellipses, 1) + 1;
                    %=================================================================
                    ellipses = [ellipses; C(i, :)];
                end
            end
        end
    end
end

% ================================================================================================================================
%
% ================================================================================================================================
function [mylabels,labels, ellipses, validCandidates] = subEllipseDetection( list, points, normals, distance_tolerance, normal_tolerance, Tmin, angleCoverage,E,angleLoop)
    labels = zeros(size(points, 1), 1);
    mylabels = zeros(size(points, 1), 1);
    ellipses = zeros(0, 5);
    max_dis = max(points) - min(points);
    maxSemiMajor = max(max_dis);
    maxSemiMinor = min(max_dis);
    distance_tolerance_square = distance_tolerance*distance_tolerance;
    validCandidates = true(size(list, 1), 1);%logical向量，大小 candidate_n x 1
    convergence = list;%候选椭圆副本
    for i = 1 : size(list, 1)
        ellipseCenter = list(i, 1 : 2);
        ellipseAxes = list(i, 3:4);
        ellipsePhi  = list(i,5);
        %ellipse circumference is approximate pi * (1.5*sum(ellipseAxes)-sqrt(ellipseAxes(1)*ellipseAxes(2))
        tbins = min( [ 180, floor( pi * (1.5*sum(ellipseAxes)-sqrt(ellipseAxes(1)*ellipseAxes(2)) ) * Tmin ) ] );
        i_dx = find( points(:,1) >= (ellipseCenter(1)-ellipseAxes(1)-distance_tolerance) & points(:,1) <= (ellipseCenter(1)+ellipseAxes(1)+distance_tolerance) & points(:,2) >= (ellipseCenter(2)-ellipseAxes(1)-distance_tolerance) & points(:,2) <= (ellipseCenter(2)+ellipseAxes(1)+distance_tolerance));
        inliers = i_dx(labels(i_dx) == 0 & (dRosin_square(list(i,:),points(i_dx,:)) <= distance_tolerance_square) );
        ellipse_normals = computePointAngle(list(i,:),points(inliers,:));

        p_dot_temp = dot(normals(inliers,:), ellipse_normals, 2);
        p_cnt = sum(p_dot_temp>0);
        if(p_cnt > size(inliers,1)*0.5)
            ellipse_polarity = -1;
            inliers = inliers(p_dot_temp>0 & p_dot_temp >= 0.923879532511287 );
        else
            ellipse_polarity = 1;
            inliers = inliers(p_dot_temp<0 & (-p_dot_temp) >= 0.923879532511287 );
        end
        inliers3 = 0;
        %=================================================================================================================================================================
        inliers = inliers(takeInliers(points(inliers, :), ellipseCenter, tbins));
         [new_ellipse,new_info] = fitEllipse(points(inliers,1),points(inliers,2));

        if (new_info == 1)
            if ( (((new_ellipse(1) - ellipseCenter(1))^2 + (new_ellipse(2) - ellipseCenter(2))^2 ) <= 16 * distance_tolerance_square) ...
                && (((new_ellipse(3) - ellipseAxes(1))^2 + (new_ellipse(4) - ellipseAxes(2))^2 ) <= 16 * distance_tolerance_square) ...
                && (abs(new_ellipse(5) - ellipsePhi) <= 0.314159265358979) )
                i_dx = find( points(:,1) >= (new_ellipse(1)-new_ellipse(3)-distance_tolerance) & points(:,1) <= (new_ellipse(1)+new_ellipse(3)+distance_tolerance) & points(:,2) >= (new_ellipse(2)-new_ellipse(3)-distance_tolerance) & points(:,2) <= (new_ellipse(2)+new_ellipse(3)+distance_tolerance));
                ellipse_normals = computePointAngle(new_ellipse,points(i_dx,:));
                newinliers = i_dx(labels(i_dx) == 0 & (dRosin_square(new_ellipse,points(i_dx,:)) <= distance_tolerance_square & ((dot(normals(i_dx,:), ellipse_normals, 2)*(-ellipse_polarity)) >= 0.923879532511287) ) );
                newinliers = newinliers(takeInliers(points(newinliers, :), new_ellipse(1:2), tbins));
                if (length(newinliers) > length(inliers))
                    inliers = newinliers;
                    inliers3 = newinliers;
                    %======================================================================
                    [new_new_ellipse,new_new_info] = fitEllipse(points(inliers,1),points(inliers,2));
                    if(new_new_info == 1)
                       new_ellipse = new_new_ellipse;
                    end
                    %=======================================================================
                end
            end
        else
            new_ellipse = list(i,:);  %candidates
        end
        if (length(inliers) >= floor( pi * (1.5*sum(new_ellipse(3:4))-sqrt(new_ellipse(3)*new_ellipse(4))) * Tmin ))
            convergence(i, :) = new_ellipse;
            if (any( (sqrt(sum((convergence(1 : i - 1, 1 : 2) - repmat(new_ellipse(1:2), i - 1, 1)) .^ 2, 2)) <= distance_tolerance) ...
                & (sqrt(sum((convergence(1 : i - 1, 3 : 4) - repmat(new_ellipse(3:4), i - 1, 1)) .^ 2, 2)) <= distance_tolerance) ...
                & (abs(convergence(1 : i - 1, 5) - repmat(new_ellipse(5), i - 1, 1)) <= 0.314159265358979) ))
                validCandidates(i) = false;
            end
            completeOrNot = calcuCompleteness(points(inliers,:),new_ellipse(1:2),tbins) >= angleCoverage;
            if (new_info == 1 && new_ellipse(3) < maxSemiMajor && new_ellipse(4) < maxSemiMinor && completeOrNot )
                if (all( (sqrt(sum((ellipses(:, 1 : 2) - repmat(new_ellipse(1:2), size(ellipses, 1), 1)) .^ 2, 2)) > distance_tolerance) ...
                   | (sqrt(sum((ellipses(:, 3 : 4) - repmat(new_ellipse(3:4), size(ellipses, 1), 1)) .^ 2, 2)) > distance_tolerance) ...
                   | (abs(ellipses(:, 5) - repmat(new_ellipse(5), size(ellipses, 1), 1)) >= 0.314159265358979 ) )) %0.1 * pi = 0.314159265358979 = 18°
                         labels(inliers) = size(ellipses, 1) + 1;
                         %==================================================================
                         if(all(inliers3) == 1)
                         mylabels(inliers3) = size(ellipses,1) + 1;
                         end
                         %==================================================================
                        ellipses = [ellipses; new_ellipse];
                        validCandidates(i) = false;
                end
            end
        else
            validCandidates(i) = false;
        end
    end %for
end%fun

% ================================================================================================================================
%
% ================================================================================================================================
function [result, longest_inliers] = isComplete(x, center, tbins, angleCoverage)
    [theta, ~] = cart2pol(x(:, 1) - center(1), x(:, 2) - center(2));
    tmin = -pi; tmax = pi;
    tt = round((theta - tmin) / (tmax - tmin) * tbins + 0.5);
    tt(tt < 1) = 1; tt(tt > tbins) = tbins;
    h = histc(tt, 1 : tbins);
    longest_run = 0;
    start_idx = 1;
    end_idx = 1;
    while (start_idx <= tbins)
        if (h(start_idx) > 0)
            end_idx = start_idx;
            while (start_idx <= tbins && h(start_idx) > 0)
                start_idx = start_idx + 1;
            end
            inliers = [end_idx, start_idx - 1];
            inliers = find(tt >= inliers(1) & tt <= inliers(2));
            run = max(theta(inliers)) - min(theta(inliers));
            if (longest_run < run)
                longest_run = run;
                longest_inliers = inliers;
            end
        end
        start_idx = start_idx + 1;
    end
    if (h(1) > 0 && h(tbins) > 0)
        start_idx = 1;
        while (start_idx < tbins && h(start_idx) > 0)
            start_idx = start_idx + 1;
        end
        end_idx = tbins;
        while (end_idx > 1 && end_idx > start_idx && h(end_idx) > 0)
            end_idx = end_idx - 1;
        end
        inliers = [start_idx - 1, end_idx + 1];
        run = max(theta(tt <= inliers(1)) + 2 * pi) - min(theta(tt >= inliers(2)));
        inliers = find(tt <= inliers(1) | tt >= inliers(2));
        if (longest_run < run)
            longest_run = run;
            longest_inliers = inliers;
        end
    end
    longest_run_deg = radtodeg(longest_run);
    h_greatthanzero_num = sum(h>0);
    result =  longest_run_deg >= angleCoverage || h_greatthanzero_num * (360 / tbins) >= min([360, 1.2*angleCoverage]);  %1.2 * angleCoverage
end
function [completeness] = calcuCompleteness(x, center, tbins)
    [theta, ~] = cart2pol(x(:, 1) - center(1), x(:, 2) - center(2));
    tmin = -pi; tmax = pi;
    tt = round((theta - tmin) / (tmax - tmin) * (tbins-1) + 1);
    tt(tt < 1) = 1; tt(tt > tbins) = tbins;
    h = histc(tt, 1 : tbins);
    h_greatthanzero_num = sum(h>0);
    completeness = h_greatthanzero_num*(360 / tbins);
end
% ================================================================================================================================
%
% ================================================================================================================================
function idx = takeInliers(x, center, tbins)
   [theta, ~] = cart2pol(x(:, 1) - center(1), x(:, 2) - center(2));
    tmin = -pi; tmax = pi;
    tt = round((theta - tmin) / (tmax - tmin) * (tbins-1) + 1);
    tt(tt < 1) = 1; tt(tt > tbins) = tbins;
    h = histc(tt, 1 : tbins);
    mark = zeros(tbins, 1);
    compSize = zeros(tbins, 1);
    nComps = 0;
    queue = zeros(tbins, 1);
    du = [-1, 1];
    for i = 1 : tbins
        if (h(i) > 0 && mark(i) == 0)
            nComps = nComps + 1;
            mark(i) = nComps;
            front = 1; rear = 1;
            queue(front) = i;
            while (front <= rear)
                u = queue(front);
                front = front + 1;
                for j = 1 : 2
                    v = u + du(j);
                    if (v == 0)
                        v = tbins;
                    end
                    if (v > tbins)
                        v = 1;
                    end
                    if (mark(v) == 0 && h(v) > 0)
                        rear = rear + 1;
                        queue(rear) = v;
                        mark(v) = nComps;
                    end
                end
            end
            compSize(nComps) = sum(ismember(tt, find(mark == nComps)));
        end
    end
    compSize(nComps + 1 : end) = [];
    maxCompSize = max(compSize);
    validComps = find(compSize > 0);
    validBins = find(ismember(mark, validComps));
    idx = ismember(tt, validBins);
end

% ================================================================================================================================
% compute the points' normals belong to an ellipse, the normals have been already normalized.
% param: [x0 y0 a b phi].
% points: [xi yi], n x 2
% ================================================================================================================================
function [ellipse_normals] = computePointAngle(ellipse, points)
%convert [x0 y0 a b phi] to Ax^2+Bxy+Cy^2+Dx+Ey+F = 0
a_square = ellipse(3)^2;
b_square = ellipse(4)^2;
sin_phi = sin(ellipse(5));
cos_phi = cos(ellipse(5));
sin_square = sin_phi^2;
cos_square = cos_phi^2;
A = b_square*cos_square+a_square*sin_square;
B = (b_square-a_square)*sin_phi*cos_phi*2;
C = b_square*sin_square+a_square*cos_square;
D = -2*A*ellipse(1)-B*ellipse(2);
E = -2*C*ellipse(2)-B*ellipse(1);
%calculate points' normals to ellipse
angles = atan2(C*points(:,2)+B/2*points(:,1)+E/2, A*points(:,1)+B/2*points(:,2)+D/2);
ellipse_normals = [cos(angles),sin(angles)];
end

% ================================================================================================================================
% square of point to ellipse distance
% ================================================================================================================================
function [dmin]= dRosin_square(param,points)
ae2 = param(3).*param(3);
be2 = param(4).*param(4);
x = points(:,1) - param(1);
y = points(:,2) - param(2);
xp = x*cos(-param(5))-y*sin(-param(5));
yp = x*sin(-param(5))+y*cos(-param(5));
fe2 = ae2-be2;
X = xp.*xp;
Y = yp.*yp;
delta = (X+Y+fe2).^2-4*fe2*X;
A = (X+Y+fe2-sqrt(delta))/2;
ah = sqrt(A);
bh2 = fe2-A;
term = A*be2+ae2*bh2;
xi = ah.*sqrt(ae2*(be2+bh2)./term);
yi = param(4)*sqrt(bh2.*(ae2-A)./term);
d = zeros(size(points,1),4);%n x 4
d(:,1) = (xp-xi).^2+(yp-yi).^2;
d(:,2) = (xp-xi).^2+(yp+yi).^2;
d(:,3) = (xp+xi).^2+(yp-yi).^2;
d(:,4) = (xp+xi).^2+(yp+yi).^2;
dmin = min(d,[],2);
end
