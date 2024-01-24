clear all; close all; clc
e = referenceEllipsoid('WGS84','km');
load('CAv1.mat')
load('../updated/CA_keras_seas_slopes.mat','llon_1','llon_2')
mop_reg = nan(size(llon_1_dat,1),1);
mop_num = nan(size(llon_1_dat,1),1);

for i = 1:size(llon_1_dat,1)
    dist = distance(CAv1.bbeach_lat,CAv1.bbeach_lon,llon_1_dat(i,2),llon_1_dat(i,1),e);
    [~,idx_min] = min(dist);
    
    label = char(CAv1.label(idx_min));
    
    if strcmp(label(1),'D') && ~isnan(str2double(label(2)))
        mop_reg(i) = 1;
        mop_num(i) = str2double(label(2:5));
    elseif strcmp(label(1:2),'OC')
        mop_reg(i) = 2;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1),'L') && ~isnan(str2double(label(2)))
        mop_reg(i) = 3;
        mop_num(i) = str2double(label(2:5));
    elseif strcmp(label(1:2),'VE')
        mop_reg(i) = 4;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'B') && ~isnan(str2double(label(2)))
        mop_reg(i) = 5;
        mop_num(i) = str2double(label(2:5));
    elseif strcmp(label(1:2),'SL')
        mop_reg(i) = 6;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'MO')
        mop_reg(i) = 7;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'SC')
        mop_reg(i) = 8;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'SM')
        mop_reg(i) = 9;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'SF')
        mop_reg(i) = 10;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'MA')
        mop_reg(i) = 11;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'SN')
        mop_reg(i) = 12;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'M') && ~isnan(str2double(label(2)))
        mop_reg(i) = 13;
        mop_num(i) = str2double(label(2:5));
    elseif strcmp(label(1:2),'HU')
        mop_reg(i) = 14;
        mop_num(i) = str2double(label(3:5));
    elseif strcmp(label(1:2),'DN')
        mop_reg(i) = 15;
        mop_num(i) = str2double(label(3:5));
    end
end
%%
mop_num = mop_num';
mop_reg = mop_reg';
save('../updated/CA_locations.mat','mop_num','mop_reg','-append')
