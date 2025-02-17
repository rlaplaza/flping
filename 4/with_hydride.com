%nprocshared=24
%mem=24GB
%chk=with_hydride.chk
#p opt freq=noraman def2svp pbe1pbe em=gd3bj integral=ultrafinegrid

with_hydride

-1 1
C        2.32353614       2.33203536       2.97451589      
C        2.14260430       1.78137847       1.58861163      
C        2.50567013       2.54450335       0.46504726      
C        2.27826260       2.01086082      -0.81040730      
C        1.69747662       0.72824720      -0.97971518      
C        1.43498912       0.19509369      -2.34002953      
C        1.30867676       0.01729596       0.20928149      
C        0.63521024      -1.28262372       0.09557852      
C        1.39985717      -2.46647477       0.17328506      
C        2.90533798      -2.42292539       0.35742206      
C        0.76384893      -3.69142222       0.07691803      
C       -0.62100211      -3.78179455      -0.09350444      
C       -1.39602036      -2.65113770      -0.14208086      
C       -2.89096339      -2.78459833      -0.26703232      
C       -0.77243048      -1.36263204      -0.07463722      
B       -1.63084944      -0.02443439      -0.13889278      
C       -2.42269782       0.25313760      -1.45130916      
C       -3.01261082       1.40998578      -1.79386024      
C       -2.95687000       2.72159580      -1.10406718      
C       -2.63726312       2.99010463       0.18139345      
C       -2.22208374       2.07335441       1.24198065      
C       -1.77122207       0.83829526       1.16330840      
N        1.54466390       0.57646533       1.42849436      
H        1.56006880       3.13119996       3.17321137      
H        2.19400674       1.54703744       3.70011355      
H        3.34651303       2.76218047       3.06744093      
H        2.93460561       3.49837431       0.58073828      
H        2.55684013       2.61861594      -1.67582370      
H        1.73783406      -0.86546457      -2.39281898      
H        2.02179851       0.75996240      -3.10358433      
H        0.36168978       0.29898945      -2.59010353      
H        3.35895838      -1.88382908      -0.50988715      
H        3.15458020      -1.88306445       1.30266013      
H        3.34339013      -3.44798171       0.40600455      
H        1.34082085      -4.63525377       0.11792114      
H       -1.06509249      -4.74975814      -0.15376722      
H       -3.21891950      -3.79898646       0.04941129      
H       -3.18571714      -2.66873870      -1.30694758      
H       -3.43452261      -2.04656353       0.35509688      
H       -2.48295778      -0.52844377      -2.16266667      
H       -3.48980468       1.42562213      -2.75277006      
H       -3.31303151       3.56087357      -1.66893544      
H       -2.74110236       4.01755802       0.47410151      
H       -2.21103876       2.49553437       2.24568163      
H       -1.42436004       0.35540892       2.04845076      
H       -1.25847708       0.97406116      -0.41155644      

--Link1--
%nprocshared=24
%mem=24GB
%chk=with_hydride.chk
#p def2tzvp pbe1pbe integral=ultrafinegrid geom=allcheck guess=read em=gd3bj

