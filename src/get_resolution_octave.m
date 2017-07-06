function resolution = get_resolution_octave( slc_sar, sample_m, line_m, RCS, OSF, paz, prg, incang)
    % Get resolution of SAR image (get basic parameters of IRF)
    % Author: Tomas Zajc (CONAE: National Commssion for Space Activities - Argentina)
	% email: tzajc@conae.gov.ar
	% Parameters:
    % -----------
    % slc_sar: complex - SAR image of NxM, where N is the number of samples in azimuth and M is the number of samples in range.
    % sample: float - pixel position of target in range.
    % line: float - pixel position of target in azimuth.
    % RCS: float - Radar Cross Section of target (in dB, typical value: 38 for CONAE corner reflectors). See http://www.rfcafe.com/references/electrical/ew-radar-handbook/images/imgp76.gif
    % OSF: float - OverSamplig Factor. (typical value: 16) 
    % paz: float - pixel size in azimuth. Theoretically is v/PRF, where v is the platform velocity and PRF is the Pulse Repetition Frequency.
    % prg: float - pixel size in range. Theoretically is c/(2*rsf), where c is speed of light and rsf is the range resampling frequency.
    % incang: float - incidence angle of target.
	% Returns:
    % --------
    % resolution: array containing:
    %             rgr: float - range resolution (-3dB).
    %             azr: float - azimuth resolution (-3dB).
    %             K: float - absolute calibration constant. Calibrate image with 10*log10(abs(slc).^2) - K.

	% Check size of image. Extend in case of too small image (zero padding).
    if ( (columns(slc_sar)<200) && (rows(slc_sar)<200) )
        l=400;
    	s=400;
	    slc = zeros(l,s);
	    s_in=size(slc_sar,2); %x: 31
	    l_in=size(slc_sar,1); %y: 55
	    % DEBUG: slc es 31x55, donde 31 es en x y 55 en y (rango vs azimuth)
	    slc(l/2-floor(l_in/2) +1:l/2+floor(l_in/2)+1, s/2-floor(s_in/2) +1:s/2+floor(s_in/2)+1)=slc_sar;
	else
		slc=slc_sar;
    end

    aa=8;
    ar=4; 
    RCS=10^(RCS/10);
    
    slc=slc(line_m-120:line_m+120, sample_m-20:sample_m+20);
    
    % Oversampling.
    [lines, samples]=size(slc);
    slc=interpft(slc, double(OSF*lines), 1);
    slc=interpft(slc, double(OSF*samples), 2);
    
    % Get power and update samples and lines.
    pow=abs(slc).^2;
    [lines, samples]=size(pow);
    figure();
    imagesc(10*log10(pow));
    title('OS IRF')
    
    % Get maximum value.
    [line_m, sample_m] = find(pow == max(pow(:)));
    
    % Compute calibration constant through integration method.
    % To compute target power we need to integrate total power and substract clutter contribution.
    p_tot = sum(pow(:));
    p_cl = sum(sum(pow(1: line_m-aa, 1:sample_m-ar)))...
          +sum(sum(pow(1: line_m-aa, sample_m+ar:end)))...
          +sum(sum(pow(line_m+aa:end, 1:sample_m-ar)))...
          +sum(sum(pow(line_m+aa:end, sample_m+ar:end)));
    
    p_cl=p_cl*numel(pow)/(numel(pow)-(ar*lines+aa*samples-ar*aa));
    pot=p_tot-p_cl;
    pazrg=paz*prg/OSF^2;
    K=pazrg*pot/(RCS*sind(incang)); 
    K=10*log10(K);
    
    % Compute resolution.
    irf_rg=pow(line_m, :);
    irf_az=pow(:, sample_m);
    irf_rg=irf_rg/max(irf_rg);
    irf_az=irf_az/max(irf_az);
    
    figure()
    plot(10*log10(irf_rg));
    title('RG IRF')
    figure()
    plot(10*log10(irf_az));
    title('AZ IRF')
    
    [~, pico_rg]=max(irf_rg);
    [~, pico_az]=max(irf_az);
    
    [~, rgr]=min(abs(10*log10(irf_rg)+3));
    [~, azr]=min(abs(10*log10(irf_az)+3));
    
    rgr=abs(2*(rgr-pico_rg)*prg/(OSF));
    azr=abs(2*(azr-pico_az)*paz/(OSF));
    
	% Return resolution values.
    resolution=[rgr, azr, K];

%endfunction
