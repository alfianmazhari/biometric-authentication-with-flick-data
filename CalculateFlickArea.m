function [ flickArea ] = CalculateFlickArea( rawx, rawy, size )
%CALCULATEFLICKAREA Summary of this function goes here
%   Detailed explanation goes here
screenArea = zeros(480,800);
pointNum = numel(rawx);

for pointCount = 1:pointNum
    pointx = rawx(pointCount);
    pointy = rawy(pointCount);
    radius = size(pointCount)*235;
    for x = floor(pointx-radius):floor(pointx+radius)
        for y = floor(pointy-radius):floor(pointy+radius)
            if( sqrt(( x-pointx)^2 + (y-pointy)^2 ) <= radius )
                if(x>0 && x<=480 && y>0 && y<= 800)
                    screenArea(x,y) = 1;
                end
            end
        end
    end
end
[u, v] = find(screenArea==1);
flickArea = numel(u);
end


