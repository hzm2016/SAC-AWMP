function [RFfoot_pos,LFfoot_pos,RBfoot_pos,LBfoot_pos]=body_cm_move(foot_pos_start,move_distance,tm,Tm)
% foot_pos_start is the position of four legs in the body center coordinate system before moving the body center
% foot_pos_start = [rf_fx,rf_fy,rf_fz;
%                              lf_fx,lf_fy,lf_fz;
%                              rb_fx,rb_fy,rb_fz;
%                              lb_fx,lb_fy,lb_fz;]
% move_distance is the displacement of body center in three directions, move_distance = [x_distance;y_distance;z_distance];
% tm is the time
% Tm is the duration of body move phase
% [RFfoot_pos,LFfoot_pos,RBfoot_pos,LBfoot_pos] is the position of four legs in body center coordinate system

%% the trajectory of body center
% the base coordinate system is the body center position before moving the body, using quintic polynomial to plan the trajectory of body center
xb0 = 0; yb0 = 0; zb0 = 0;%the body center's coordinate value before moving the body
xb_end = move_distance(1);%the distance of body move in x 
yb_end = move_distance(2);%the distance of body move in y
zb_end = move_distance(3);%the distance of body move in z

% when time is tm, the distance of body center moving 
xb = xb0 + 10*(xb_end-xb0)/(Tm^3)*tm^3 + 15*(xb0-xb_end)/(Tm^4)*tm^4 + 6*(xb_end-xb0)/(Tm^5)*tm^5;
yb = yb0 + 10*(yb_end-yb0)/(Tm^3)*tm^3 + 15*(yb0-yb_end)/(Tm^4)*tm^4 + 6*(yb_end-yb0)/(Tm^5)*tm^5;
zb = zb0 + 10*(zb_end-zb0)/(Tm^3)*tm^3 + 15*(zb0-zb_end)/(Tm^4)*tm^4 + 6*(zb_end-zb0)/(Tm^5)*tm^5;
%% during body moving, the coordinate value of the four legs's foot in body center coordinate system
%right fore leg
RFfoot_pos(1) = foot_pos_start(1,1) - xb;
RFfoot_pos(2) = foot_pos_start(1,2) - yb;
RFfoot_pos(3) = foot_pos_start(1,3) - zb;
%left fore leg
LFfoot_pos(1) = foot_pos_start(2,1) - xb;
LFfoot_pos(2) = foot_pos_start(2,2) - yb;
LFfoot_pos(3) = foot_pos_start(2,3) - zb;
%right hind leg
RBfoot_pos(1) = foot_pos_start(3,1) - xb;
RBfoot_pos(2) = foot_pos_start(3,2) - yb;
RBfoot_pos(3) = foot_pos_start(3,3) - zb;
%left hind leg
LBfoot_pos(1) = foot_pos_start(4,1) - xb;
LBfoot_pos(2) = foot_pos_start(4,2) - yb;
LBfoot_pos(3) = foot_pos_start(4,3) - zb;
end