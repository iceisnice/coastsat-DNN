clear all; close all; clc
% Adjusted California coast bounds in the 0 to 360 longitude range
lat_min = 31;     % Southernmost point of California
lat_max = 43;     % Northernmost point of California
lon_min = 360 - 125.5; % Westernmost point of California, converted to 0-360 range
lon_max = 360 - 114.1; % Easternmost point of California, converted to 0-360 range

% Calculating dimensions based on 0.25 degree resolution
lat_dim = round((lat_max - lat_min) / 0.25);
lon_dim = round((lon_max - lon_min) / 0.25);
time_dim = 25 * 12; % Total months from Jan 2000 to Dec 2023

% Initialize the 3D matrix to store the data
sla_matrix = NaN(lat_dim, lon_dim, time_dim);

% Initialize the time index vector
time_index = datetime(1999, 1, 15):calmonths(1):datetime(2023, 12, 15);
time_index = time_index';

% Loop through each file until July 2022
for year = 1999:2022
    for month = 1:12
        % Skip processing for months after July 2022
        if year == 2022 && month > 7
            break;
        end

        % Construct the file name
        filename = sprintf('data/dt_global_twosat_phy_l4_%04d%02d_vDT2021-M01.nc', year, month);

        % Read latitude, longitude, and sla data
        latitude = ncread(filename, 'latitude');
        longitude = ncread(filename, 'longitude');
        sla = ncread(filename, 'sla');

        % Find indices for California coast
        lat_indices = find(latitude >= lat_min & latitude <= lat_max);
        lon_indices = find(longitude >= lon_min & longitude <= lon_max);

        % Subset the sla data
        sla_subset = sla(lon_indices, lat_indices);
        % Store in the matrix
        time_index_mat = (year - 1999) * 12 + month;
        sla_matrix(:, :, time_index_mat) = sla_subset';
    end
end

% Fill data from August 2022 to December 2023 with zeros
sla_matrix(:, :, (2022 - 1999) * 12 + 8:end) = 0;

lat_sla = latitude(lat_indices);
lon_sla = longitude(lon_indices);

time_sla = time_index;
save('sla_CA_subset','sla_matrix','lat_sla','lon_sla','time_sla')
