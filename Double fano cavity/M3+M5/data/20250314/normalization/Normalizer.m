% Backgroud laser off measurement
fileno=4;
File0=[num2str(fileno,'%03.0f'),'.mat'];

% Background laser on measurement (no mirror or membrane)
fileno=2;
File1=[num2str(fileno,'%03.0f'),'.mat'];

Reflection_measured=1;  % If refletion is also measured, set this to 1, otherwise 0 
% Backgroud laser on measurement (high reflective mirror (R=100%) installed)
fileno=3;
HR_Reflection=0.98;
if Reflection_measured
    File2=[num2str(fileno,'%03.0f'),'.mat'];	
else
    File2=File1;
end
 



% Main measurement (membrane installed)
fileno=1;
File3=[num2str(fileno,'%03.0f'),'.mat'];

F0=load(File0,'V','wavelengths');
F1=load(File1,'V','wavelengths');
F2=load(File2,'V','wavelengths');	
F3=load(File3,'V','wavelengths');

V0=F0.V;
V1=F1.V;
V2=F2.V;
V3=F3.V;
Lambda0=F0.wavelengths';
Lambda1=F1.wavelengths';
Lambda2=F2.wavelengths';
Lambda=F3.wavelengths';

Pt0=min(V0(:,1),100);
Pt1=min(V1(:,1),100);
Pt3=min(V3(:,1),100);
Pi0=min(V0(:,2),100);
Pi1=min(V1(:,2),100);
Pi2=min(V2(:,2),100);
Pi3=min(V3(:,2),100);
Pr0=min(V0(:,3),100);
Pr2=min(V2(:,3),100)/HR_Reflection;
Pr3=min(V3(:,3),100);

Pt0_interpolated = interp1(Lambda0,Pt0,Lambda);
Pt1_interpolated = interp1(Lambda1,Pt1,Lambda);
Pi0_interpolated = interp1(Lambda0,Pi0,Lambda);
Pi1_interpolated = interp1(Lambda1,Pi1,Lambda);
Pi2_interpolated = interp1(Lambda2,Pi2,Lambda);
Pr0_interpolated = interp1(Lambda0,Pr0,Lambda);
Pr2_interpolated = interp1(Lambda2,Pr2,Lambda);


Pt=(Pt3-Pt0_interpolated)./(Pt1_interpolated-Pt0_interpolated);
Pit=(Pi3-Pi0_interpolated)./(Pi1_interpolated-Pi0_interpolated);
Pr=(Pr3-Pr0_interpolated)./(Pr2_interpolated-Pr0_interpolated);
Pir=(Pi3-Pi0_interpolated)./(Pi2_interpolated-Pi0_interpolated);


figure(103)
plot(Lambda,Pt,'-b');hold on
plot(Lambda,Pit,'-c');
plot(Lambda,Pr,'-r');
plot(Lambda,Pir,'-m');hold off
legend ('Pt','Pit')
legend ('Pt','Pit','Pr','Pir')


T=Pt./Pit;
R=Pr./Pir;

fig=figure(102)
plot(Lambda,T','-b');hold on
if Reflection_measured	plot(Lambda,R','-r');  end
if Reflection_measured	plot(Lambda,(1-T-R)','-g');  end
axis([-inf inf 0 1])
hold off
legend ('T','R','L')
% New code for saving wavelengths and voltages to a text file
data_to_save = [Lambda, T(:, 1)]; % Assuming V(:, 1) contains the voltage data you want to save
txt_filename = [filename, '.txt'];
writematrix(data_to_save, txt_filename, 'Delimiter', '\t');
% Save everything
formatOut = 'yyyymmdd_HH_MM_SS';
filename=num2str([datestr(now,formatOut),'_NormilizedTransmission_N16']);
save(filename);
savefig([filename,'.fig'])
saveas(fig,[filename,'.png']);
% New code for saving wavelengths and voltages to a text file
data_to_save = [Lambda, R(:, 1)]; % Assuming V(:, 1) contains the voltage data you want to save
txt_filename = [filename, '.txt'];
writematrix(data_to_save, txt_filename, 'Delimiter', '\t');