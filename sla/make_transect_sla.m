clear all; close all; clc
load('sla_CA_subset','sla_matrix','lat_sla','lon_sla','time_sla')
load('../CA_keras_seas_slopes.mat','llon_2')
lon_transect = llon_2(:,1);
lat_transect = llon_2(:,2);
lon_transect=mod(lon_transect,360);

[lon_sla,lat_sla] = meshgrid(lon_sla,lat_sla);
lat_sla_nonan = lat_sla(:);
lon_sla_nonan = lon_sla(:);
sla_nonan = squeeze(sla_matrix(:,:,1));
sla_nonan = sla_nonan(:);
lat_sla_nonan = lat_sla_nonan(~isnan(sla_nonan));
lon_sla_nonan = lon_sla_nonan(~isnan(sla_nonan));

sla_transect = zeros(length(lon_transect),length(time_sla));

for i = 1:length(lon_transect)
    [arclen,az] = distance(lat_sla_nonan,lon_sla_nonan,lat_transect(i),lon_transect(i),wgs84Ellipsoid);
    [~,idx] = min(arclen);
    [row, col] = find(lat_sla == lat_sla_nonan(idx) & lon_sla == lon_sla_nonan(idx));
    sla_transect(i,:) = squeeze(sla_matrix(row,col,:));
end

time_sla = datenum(time_sla);
save('sla_transect','sla_transect','time_sla')






