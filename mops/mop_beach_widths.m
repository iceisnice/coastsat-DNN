clear all; close all; clc
e = referenceEllipsoid('WGS84','km');
load('CAv1.mat')
load('../CA_seas_slopes.mat')
load('MOP_BeachWidths.mat')
tmin = datenum('2000-01-01');
MOP_id(time<=tmin) = NaN;

for i = 1:7253
    dist = distance(CAv1.bbeach_lat,CAv1.bbeach_lon,llon_1(i,2),llon_1(i,1),e);
    [~,idx_min] = min(dist);
    dist_mop = dist(idx_min);
    if dist_mop*1e3<200
        idx = MOP_id==idx_min;
        if sum(idx)>0
        time_width = time(idx);
        msl_width = msl(idx);
        save(['MOP_Beach_Widths/','MOP_Width_',num2str(i),'.mat'],'dist_mop','time_width','msl_width')
        end
    end
    i
end