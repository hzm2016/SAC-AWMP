function dog_cfg()
% dog's size parameter and motion parameter(s,m)
global dog;

% size
dog.Lhx = 0.35;% body
dog.Lhy = 0.165;
dog.Hb = 0.06;
dog.L1 = 0.3;% leg
dog.L2 = 0.3;
dog.Hr = 0.025;%foot

% motion parameter
dog.Tstable = 0.5;
dog.Tsqu_up = 0.5;
dog.Tadjust_swing = 0.5;
dog.Tsta_adjust = 0.5;
dog.Tadjust = dog.Tstable + dog.Tsqu_up + 2*dog.Tadjust_swing + 2*dog.Tsta_adjust;

dog.Tswing = 0.5;
dog.Tmove = 0.5;
dog.Tsta_gait = 0.5;
dog.Tgait = 4*dog.Tswing + 2*dog.Tmove + 2*dog.Tsta_gait;% the duration of straight intermitent crawl gait

% the parameter of leg swing and body moving
dog.Xswing = 0.40;
dog.Yswing = 0;
dog.Zswing = 0.080;

dog.Xbody_move = dog.Xswing/2;
dog.Ybody_move = 0;
dog.Zbody_move = 0.150;

% the initial angle of every joint
dog.RFLeg_q0 = [-pi/3; 2*pi/3];
dog.LFLeg_q0 = [-pi/3; 2*pi/3];
dog.RBLeg_q0 = [ pi/3;-2*pi/3];
dog.LBLeg_q0 = [pi/3;-2*pi/3];

% the desired angle of every joint
dog.RFLegq_d = [-pi/3; 2*pi/3];
dog.LFLegq_d = [-pi/3; 2*pi/3];
dog.RBLegq_d = [ pi/3;-2*pi/3];
dog.LBLegq_d = [ pi/3;-2*pi/3];

% the command of every joint
dog.RFq_cmd = [0;0];
dog.LFq_cmd = [0;0];
dog.RBq_cmd = [0;0];
dog.LBq_cmd = [0;0];

% when t = 0, the coordinate value of four leg's foot in body center coordination
dog.RFfoot_pos_in_body = [ dog.Lhx;    -dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];
dog.LFfoot_pos_in_body = [ dog.Lhx;     dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];
dog.RBfoot_pos_in_body = [-dog.Lhx;   -dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];
dog.LBfoot_pos_in_body = [-dog.Lhx;    dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];

dog.RFfoot_pos_in_body_d  = [ dog.Lhx;    -dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];
dog.LFfoot_pos_in_body_d  = [ dog.Lhx;     dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];
dog.RBfoot_pos_in_body_d = [-dog.Lhx;    -dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];
dog.LBfoot_pos_in_body_d = [-dog.Lhx;     dog.Lhy;-dog.Hb/2-dog.L1-dog.Hr];

dog.foot_pos_standup_start = [ dog.Lhx,     -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr;...%stand up phase
                                                   dog.Lhx,      dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr;...
                                                  -dog.Lhx,    -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr;...
                                                  -dog.Lhx,     dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr];
dog.foot_pos_adjust_swinglb_start = [ dog.Lhx,    -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before swing left hind leg
                                                              dog.Lhx,     dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                             -dog.Lhx,   -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                             -dog.Lhx,    dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
dog.foot_pos_adjust_swinglf_start = [ dog.Lhx,                               -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before swing left fore leg
                                                              dog.Lhx,                                dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                             -dog.Lhx,                              -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                             -dog.Lhx+dog.Xswing/2,       dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];

dog.foot_pos_gait_bodymove1_start = [  dog.Lhx,                           -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before moving body center
                                                                  dog.Lhx+dog.Xswing/2,    dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                                -dog.Lhx,                            -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                                -dog.Lhx+dog.Xswing/2,     dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
dog.foot_pos_gait_swingrb_start = [  dog.Lhx-dog.Xbody_move,                            -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before swing right hind leg
                                                            dog.Lhx+dog.Xswing/2-dog.Xbody_move,     dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                           -dog.Lhx-dog.Xbody_move,                            -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                           -dog.Lhx+dog.Xswing/2-dog.Xbody_move,     dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
dog.foot_pos_gait_swingrf_start = [  dog.Lhx-dog.Xbody_move,                           -dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before swing right fore leg
                                                           dog.Lhx+dog.Xswing/2-dog.Xbody_move,    dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                          -dog.Lhx-dog.Xbody_move+dog.Xswing,      -dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                          -dog.Lhx+dog.Xswing/2-dog.Xbody_move,    dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
dog.foot_pos_gait_bodymove2_start = [  dog.Lhx-dog.Xbody_move+dog.Xswing,     -dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before moving body center
                                                                  dog.Lhx+dog.Xswing/2-dog.Xbody_move,    dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                                -dog.Lhx-dog.Xbody_move+dog.Xswing,      -dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                                -dog.Lhx+dog.Xswing/2-dog.Xbody_move,    dog.Lhy,     -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
dog.foot_pos_gait_swinglb_start = [  dog.Lhx-dog.Xbody_move+dog.Xswing-dog.Xbody_move,       -dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before swing left hind leg
                                                            dog.Lhx+dog.Xswing/2-dog.Xbody_move-dog.Xbody_move,     dog.Lhy,    -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                           -dog.Lhx-dog.Xbody_move+dog.Xswing-dog.Xbody_move,        -dog.Lhy,  -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                          -dog.Lhx+dog.Xswing/2-dog.Xbody_move-dog.Xbody_move,       dog.Lhy,   -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
dog.foot_pos_gait_swinglf_start = [  dog.Lhx-dog.Xbody_move+dog.Xswing-dog.Xbody_move,                           -dog.Lhy,      -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...%before swing left fore leg
                                                           dog.Lhx+dog.Xswing/2-dog.Xbody_move-dog.Xbody_move,                         dog.Lhy,      -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                          -dog.Lhx-dog.Xbody_move+dog.Xswing-dog.Xbody_move,                           -dog.Lhy,      -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move;...
                                                          -dog.Lhx+dog.Xswing/2-dog.Xbody_move-dog.Xbody_move+dog.Xswing,    dog.Lhy,      -dog.Hb/2-dog.L1-dog.Hr-dog.Zbody_move];
end
