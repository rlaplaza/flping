%nprocshared=24
%mem=24GB
%chk=neutral.chk
#p opt freq=noraman def2svp pbe1pbe em=gd3bj integral=ultrafinegrid

neutral

0 1
C       -0.23214986       2.64749536       0.87086689
C       -0.98490270       1.50327347       0.75942063
B       -0.54290687       0.15469135       0.09850326
C        0.98992760      -0.11907118      -0.19011130
C        1.44402662      -0.30389880      -1.53655918
C        0.49557712      -0.24797031      -2.69697653
C        2.79563176      -0.53102004      -1.74516451
C        3.72275424      -0.54513672      -0.72249948
C        5.17004217      -0.81540377      -0.99426794
C        3.27919598      -0.40090139       0.60361906
C        1.90890193      -0.17453622       0.88539603
C        1.48200289      -0.05695055       2.31793556
C       -1.62608336      -0.97892270      -0.28943286
C       -1.33144639      -2.31709833      -0.08194332
F       -0.15349648      -2.71999304       0.44442678
C       -2.27843044      -3.30690939      -0.41805578
C       -3.49900882      -2.95270940      -0.96993616
C       -3.80417782      -1.59396622      -1.17520424
F       -4.98048700      -1.22837720      -1.73718918
C       -2.87292812      -0.59614337      -0.82985155
F       -3.17102674       0.69585641      -1.12104135
N       -2.21358207       1.72729226       1.34540840
C       -2.11511010       3.00487233       1.76545936
C       -3.23578720       3.71072796       2.46349137
O       -0.95712526       3.56598444       1.45753155
H        0.78105503       2.79984347       0.50248493
H       -0.12439739      -1.17359769      -2.71774672
H        1.05678213      -0.16423806      -3.66744670
H       -0.16040694       0.65266743      -2.62932298
H        3.15974951      -0.67144360      -2.76899233
H        5.46847078      -0.40989755      -1.98541853
H        5.35357938      -1.91530708      -0.97666415
H        5.82049620      -0.32567445      -0.21881387
H        3.98852613      -0.46113671       1.41496055
H        2.26969485      -0.46419877       3.01548414
H        1.32020751       1.02026086       2.57906078
H        0.57061550      -0.64972646       2.50272829
H       -2.04401514      -4.34826870      -0.24693058
H       -4.22542561      -3.69408853      -1.24006612
H       -2.84139578       4.57499223       3.03789633
H       -3.74405368       3.00384613       3.15302594
H       -3.96265351       4.08879757       1.71327456

--Link1--
%nprocshared=24
%mem=24GB
%chk=neutral.chk
#p def2tzvp pbe1pbe integral=ultrafinegrid geom=allcheck guess=read em=gd3bj

