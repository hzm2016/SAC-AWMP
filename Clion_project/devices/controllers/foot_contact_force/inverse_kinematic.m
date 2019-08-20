function q = inverse_kinematics(foot_to_body, Nleg)
% foot_to_body is the coordinate value of  Nleg foot in the body center coordinate system, foot_to_body=[x,y,z]
% Nleg is the Leg number
% q = [q1;q2], the joint's angle of Nleg leg

%%
global dog;

fpx=foot_to_body(1);
fpy=foot_to_body(2);
fpz=foot_to_body(3)+dog.Hr;
switch Nleg
    case 1      %left fore leg
        xbhi = dog.Lhx;  % the origin coordinate of hip coordinate system in the body center coordinate system
        ybhi = dog.Lhy;
        zbhi =-dog.Hb/2;
        
        g1=fpx-xbhi;
        g2=fpz - zbhi;
        g3=1;
        q3=g3*acos((g1^2 + g2^2 - dog.L1^2 - dog.L2^2)/(2*dog.L1*dog.L2));
        
        g4=dog.L2*sin(q3) + g1;
        q2=2*atan((g2 + sqrt(g2^2 - g4*(dog.L2*sin(q3) - g1)))/g4);
    case 2      %right fore leg
        xbhi = dog.Lhx;  
        ybhi =-dog.Lhy;
        zbhi =-dog.Hb/2; 
        
        g1=fpx-xbhi;
        g2= fpz - zbhi;
        g3=1;
        q3=g3*acos((g1^2 + g2^2 - dog.L1^2 - dog.L2^2)/(2*dog.L1*dog.L2));
        
        g4=dog.L2*sin(q3) + g1;
        q2=2*atan((g2 + sqrt(g2^2 - g4*(dog.L2*sin(q3) - g1)))/g4);
    case 3      %left hind leg
        xbhi =-dog.Lhx;  
        ybhi = dog.Lhy;
        zbhi =-dog.Hb/2; 
        
        g1=fpx-xbhi;
        g2= fpz - zbhi;
        g3=-1;
        q3=g3*acos((g1^2 + g2^2 - dog.L1^2 - dog.L2^2)/(2*dog.L1*dog.L2));
        
        g4=dog.L2*sin(q3) + g1;
        q2=2*atan((g2 + sqrt(g2^2 - g4*(dog.L2*sin(q3) - g1)))/g4);
    
    case 4      %right hind leg
        xbhi =-dog.Lhx; 
        ybhi =-dog.Lhy;
        zbhi =-dog.Hb/2; 
        
        g1=fpx-xbhi;
        g2=fpz - zbhi;
        g3=-1;
        q3=g3*acos((g1^2 + g2^2 - dog.L1^2 - dog.L2^2)/(2*dog.L1*dog.L2));
        
        g4=dog.L2*sin(q3) + g1;
        q2=2*atan((g2 + sqrt(g2^2 - g4*(dog.L2*sin(q3) - g1)))/g4);
    otherwise
    error('Quadruped robot is only four feet');    
end

q = [q2;q3];
end