% MATLAB controller for Webots
% File:          force_contact_force
% Date:          2017.8.25
% Description:
% Author:        Albert
% Modifications:

clear all;
close all;
clc;

dog_cfg();
global dog;

TIME_STEP = 16;
t = 0.0;

%%  get the devices in the world
% joint servo
rf_hip = wb_robot_get_device('rf_leg_hip');
rf_knee = wb_robot_get_device('rf_leg_knee');
lf_hip = wb_robot_get_device('lf_leg_hip');
lf_knee = wb_robot_get_device('lf_leg_knee');
rb_hip = wb_robot_get_device('rb_leg_hip');
rb_knee = wb_robot_get_device('rb_leg_knee');
lb_hip = wb_robot_get_device('lb_leg_hip');
lb_knee = wb_robot_get_device('lb_leg_knee');
% touch sensor of every leg
rf_touch_sensor = wb_robot_get_device('rf_touch_sensor');
lf_touch_sensor = wb_robot_get_device('lf_touch_sensor');
rb_touch_sensor = wb_robot_get_device('rb_touch_sensor');
lb_touch_sensor = wb_robot_get_device('lb_touch_sensor');
% body gps / imu / gyro / acceleration
gps = wb_robot_get_device('body_gps');
imu = wb_robot_get_device('body_imu');
gyro = wb_robot_get_device('body_gyro');
accelerometer = wb_robot_get_device('body_accelerometer');

%% enable all the sensors
% joint servo
wb_servo_enable_position(rf_hip,TIME_STEP);
wb_servo_enable_position(rf_knee,TIME_STEP);
wb_servo_enable_position(lf_hip,TIME_STEP);
wb_servo_enable_position(lf_knee,TIME_STEP);
wb_servo_enable_position(rb_hip,TIME_STEP);
wb_servo_enable_position(rb_knee,TIME_STEP);
wb_servo_enable_position(lb_hip,TIME_STEP);
wb_servo_enable_position(lb_knee,TIME_STEP);
% touch sensor
wb_touch_sensor_enable(rf_touch_sensor,TIME_STEP);
wb_touch_sensor_enable(lf_touch_sensor,TIME_STEP);
wb_touch_sensor_enable(rb_touch_sensor,TIME_STEP);
wb_touch_sensor_enable(lb_touch_sensor,TIME_STEP);
% body gps / imu / gyro / acceleration
%wb_gps_enable(gps, TIME_STEP);
% wb_inertial_unit_enable(imu, TIME_STEP);
%wb_gyro_enable(gyro, TIME_STEP);
%wb_accelerometer_enable(accelerometer, TIME_STEP);


while (wb_robot_step(TIME_STEP) ~= -1)
  t_gait = t - dog.Tadjust;%t - dog.Tadjust; %t_gait is the time after the dog start walk

  %% get the foot's force
  rf_touch_value = wb_touch_sensor_get_values(rf_touch_sensor);%unit is N.
  rf_touch_total = sqrt(rf_touch_value(1)^2+rf_touch_value(2)^2+rf_touch_value(3)^2);
  lf_touch_value = wb_touch_sensor_get_values(lf_touch_sensor);
  lf_touch_total = sqrt(lf_touch_value(1)^2+lf_touch_value(2)^2+lf_touch_value(3)^2);
  rb_touch_value = wb_touch_sensor_get_values(rb_touch_sensor);
  rb_touch_total = sqrt(rb_touch_value(1)^2+rb_touch_value(2)^2+rb_touch_value(3)^2);
  lb_touch_value = wb_touch_sensor_get_values(lb_touch_sensor);
  lb_touch_total = sqrt(lb_touch_value(1)^2+lb_touch_value(2)^2+lb_touch_value(3)^2);
  disp(['t = ', num2str(t)]);
  disp([ ' rf_touch_value:[ ', num2str(rf_touch_value(1)), ' , ', num2str(rf_touch_value(2)), ' , ', num2str(rf_touch_value(3)),']', '>amplitude: ', num2str(rf_touch_total)]);
  disp([ ' lf_touch_value:[ ', num2str(lf_touch_value(1)), ' , ', num2str(lf_touch_value(2)), ' , ', num2str(lf_touch_value(3)),']', '>amplitude: ', num2str(lf_touch_total)]);
  disp([ ' rb_touch_value:[ ', num2str(rb_touch_value(1)), ' , ', num2str(rb_touch_value(2)), ' , ', num2str(rb_touch_value(3)),']', '>amplitude: ', num2str(rb_touch_total)]);
  disp([ ' lb_touch_value:[ ', num2str(lb_touch_value(1)), ' , ', num2str(lb_touch_value(2)), ' , ', num2str(lb_touch_value(3)),']', '>amplitude: ', num2str(lb_touch_total)]);
  drawnow;
  if ((t <= dog.Tstable) && (t >= 0))
        % do nothing;
  elseif ((t > dog.Tstable)) &&(t <= (dog.Tstable+dog.Tsqu_up) )
        t_standup = t - dog.Tstable;
        standup_distance = [0;0;dog.Zbody_move];
        [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=body_cm_move(dog.foot_pos_standup_start,standup_distance,t_standup,dog.Tsqu_up);
  elseif ((t > (dog.Tstable+dog.Tsqu_up))) &&(t <= (dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust) )
        % do nothing;
  elseif ((t > (dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust)) &&(t <= (dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust+dog.Tadjust_swing)) )
        ts_lb = t-(dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust);
        Nleg = 3;
        swing_lb_distance = [dog.Xswing/2;dog.Yswing;dog.Zswing];
        [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=leg_swing(Nleg,dog.foot_pos_adjust_swinglb_start,swing_lb_distance,ts_lb,dog.Tadjust_swing);
  elseif ((t > (dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust+dog.Tadjust_swing)) &&(t <= (dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust+2*dog.Tadjust_swing)) )
        ts_lf = t-(dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust+dog.Tadjust_swing);
        Nleg = 1;
        swing_lf_distance = [dog.Xswing/2;dog.Yswing;dog.Zswing];
        [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=leg_swing(Nleg,dog.foot_pos_adjust_swinglf_start,swing_lf_distance,ts_lf,dog.Tadjust_swing);
  elseif ((t > (dog.Tstable+dog.Tsqu_up+dog.Tsta_adjust+2*dog.Tadjust_swing)) &&(t <= dog.Tadjust) )
        % do nothing;
  end

  if t_gait >= 0
        tg = mod(t_gait,dog.Tgait); % tg is the time for straight intermittent crawl gait
        if ((tg >= 0) && (tg <= dog.Tmove))% foreward moving body center
              tmove_1 = tg;
              bmove_distance_1 = [dog.Xbody_move;0;0];
              [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=body_cm_move(dog.foot_pos_gait_bodymove1_start,bmove_distance_1,tmove_1,dog.Tmove);

        elseif ((tg > dog.Tmove) && (tg <= (dog.Tmove+dog.Tswing)))% swing right hind leg
              ts_rb = tg-dog.Tmove;
              Nleg = 4;
              swing_rb_distance = [dog.Xswing;dog.Yswing;dog.Zswing];
              [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=leg_swing(Nleg,dog.foot_pos_gait_swingrb_start,swing_rb_distance,ts_rb,dog.Tswing);

        elseif ((tg > (dog.Tmove+dog.Tswing)) && (tg <= (dog.Tmove+2*dog.Tswing)))% swing right fore leg
              ts_rf = tg-(dog.Tmove+dog.Tswing);
              Nleg = 2;
              swing_rf_distance = [dog.Xswing;dog.Yswing;dog.Zswing];
              [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=leg_swing(Nleg,dog.foot_pos_gait_swingrf_start,swing_rf_distance,ts_rf,dog.Tswing);

        elseif ((tg > (dog.Tmove+2*dog.Tswing)) && (tg <= (dog.Tmove+2*dog.Tswing+dog.Tsta_gait)))% measure the force of foot

        elseif ((tg > (dog.Tmove+2*dog.Tswing+dog.Tsta_gait)) && (tg <= (2*dog.Tmove+2*dog.Tswing+dog.Tsta_gait)))% forward moving body center
              tmove_2 = tg-(dog.Tmove+2*dog.Tswing+dog.Tsta_gait);
              bmove_distance_2 = [dog.Xbody_move;0;0];
              [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=body_cm_move(dog.foot_pos_gait_bodymove2_start,bmove_distance_2,tmove_2,dog.Tmove);

        elseif ((tg > (2*dog.Tmove+2*dog.Tswing+dog.Tsta_gait)) && (tg <= (2*dog.Tmove+3*dog.Tswing+dog.Tsta_gait)))% swing left hind leg
              ts_lb = tg-(2*dog.Tmove+2*dog.Tswing+dog.Tsta_gait);
              Nleg = 3;
              swing_lb_distance = [dog.Xswing;dog.Yswing;dog.Zswing];
              [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=leg_swing(Nleg,dog.foot_pos_gait_swinglb_start,swing_lb_distance,ts_lb,dog.Tswing);

        elseif ((tg > (2*dog.Tmove+3*dog.Tswing+dog.Tsta_gait)) && (tg <= (2*dog.Tmove+4*dog.Tswing+dog.Tsta_gait)))% swing right fore leg
              ts_lf = tg-(2*dog.Tmove+3*dog.Tswing+dog.Tsta_gait);
              Nleg = 1;
              swing_lf_distance = [dog.Xswing;dog.Yswing;dog.Zswing];
              [dog.RFfoot_pos_in_body_d,dog.LFfoot_pos_in_body_d,dog.RBfoot_pos_in_body_d,dog.LBfoot_pos_in_body_d]=leg_swing(Nleg,dog.foot_pos_gait_swinglf_start,swing_lf_distance,ts_lf,dog.Tswing);

       elseif ((tg > (2*dog.Tmove+4*dog.Tswing+dog.Tsta_gait)) && (tg <= dog.Tgait))% measure the force of foot
              % do nothing;
        end
  end

  %  compute the angle of every joint
  dog.RFLegq_d = inverse_kinematic(dog.RFfoot_pos_in_body_d,2);
  dog.LFLegq_d = inverse_kinematic(dog.LFfoot_pos_in_body_d,1);
  dog.RBLegq_d = inverse_kinematic(dog.RBfoot_pos_in_body_d,4);
  dog.LBLegq_d = inverse_kinematic(dog.LBfoot_pos_in_body_d,3);

  dog.RFq_cmd = dog.RFLegq_d - dog.RFLeg_q0;
  dog.LFq_cmd = dog.LFLegq_d - dog.LFLeg_q0;
  dog.RBq_cmd = dog.RBLegq_d - dog.RBLeg_q0;
  dog.LBq_cmd = dog.LBLegq_d - dog.LBLeg_q0;

  % send actuator commands, e.g.:
  wb_servo_set_position(rf_hip, dog.RFq_cmd(1));
  wb_servo_set_position(rf_knee, dog.RFq_cmd(2));
  wb_servo_set_position(lf_hip, dog.LFq_cmd(1));
  wb_servo_set_position(lf_knee, dog.LFq_cmd(2));
  wb_servo_set_position(rb_hip, dog.RBq_cmd(1));
  wb_servo_set_position(rb_knee, dog.RBq_cmd(2));
  wb_servo_set_position(lb_hip, dog.LBq_cmd(1));
  wb_servo_set_position(lb_knee, dog.LBq_cmd(2));

  t = t + TIME_STEP/1000;
 end
