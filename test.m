clc 
clear
workingDir = "/home/leelcz/Graduate_project";


imageNames = dir(fullfile(workingDir,'export','*.jpg'));

outputVideo = VideoWriter(fullfile(workingDir,'robotarm.avi'));

outputVideo.FrameRate = 30;
open(outputVideo)

for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,'export',imageNames(ii).name));
   writeVideo(outputVideo,img)
end

close(outputVideo)