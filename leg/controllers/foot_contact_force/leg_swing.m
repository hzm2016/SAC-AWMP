function [rf_foot,lf_foot,rb_foot,lb_foot]=leg_swing(Nleg,foot_pos_start,swing_distance,ts,Ts)
% Nleg is Leg number
% foot_pos_start = [rf_fx,rf_fy,rf_fz;    foot_pos_start is the position of four leg in the body center coordination before leg swing.
%                              lf_fx,lf_fy,lf_fz;
%                              rb_fx,rb_fy,rb_fz;
%                              lb_fx,lb_fy,lb_fz;]
% swing_distance is the distance of leg swing in three directions, swing_distance = [xd;yd;zd];
% ts is the time;
% Ts is the duration of leg swing

Lx = swing_distance(1);%the distance of leg swing in x
Ly = swing_distance(2);%the distance of leg swing in y
Lz = swing_distance(3);%the distance of leg swing in z
k=1;%hight coefficient

rf_foot = foot_pos_start(1,:);
lf_foot = foot_pos_start(2,:);
rb_foot = foot_pos_start(3,:);
lb_foot = foot_pos_start(4,:);

%% the foot trajectory of leg swing
%the base coordination is the foot position before leg swing
xf=0;
yf=0;
zf=0;
%Vertical lift stage
if ((ts<=Ts/4) && (ts>=0))
    xf = 0;
    yf = 0;
    zf = 4*Lz*ts/Ts;
elseif ((ts<=Ts/2) && (ts>Ts/4)) %forware swing stage
    xf = 4*Lx*(ts-Ts/4)/Ts;
    yf = 4*Ly*(ts-Ts/4)/Ts;
    zf = Lz;
elseif ((ts<=Ts) && (ts>Ts/2))%Vertical down stage
    xf = Lx;
    yf = Ly;
    zf = -k*2*Lz*(ts-Ts/2)/Ts+Lz;
end

%% the foot trajectory in the body center coordination
switch (Nleg)
    case 2 %right fore leg
        rf_foot(1) = foot_pos_start(1,1) + xf;
        rf_foot(2) = foot_pos_start(1,2) + yf;
        rf_foot(3) = foot_pos_start(1,3) + zf;
    case 1 %left fore leg
        lf_foot(1) = foot_pos_start(2,1) + xf;
        lf_foot(2) = foot_pos_start(2,2) + yf;
        lf_foot(3) = foot_pos_start(2,3) + zf;
    case 4 %right hind leg
        rb_foot(1) = foot_pos_start(3,1) + xf;
        rb_foot(2) = foot_pos_start(3,2) + yf;
        rb_foot(3) = foot_pos_start(3,3) + zf;
    case 3 %left hind leg
        lb_foot(1) = foot_pos_start(4,1) + xf;
        lb_foot(2) = foot_pos_start(4,2) + yf;
        lb_foot(3) = foot_pos_start(4,3) + zf;
end
end