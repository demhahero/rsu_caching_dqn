import matplotlib.pyplot as plt



r2=[-81.0, -40.0, -368.0, -150.0, -68.0, -51.0]
c2=[780.0, 1515.0, 2394.0, 3030.0, 3703.0, 4573.0]
s2=[0.36245785905163375, 0.3906747387433196, 0.3507963394020007, 0.3839286547198149, 0.3980570831722995, 0.3985566197967669]

u2=[1561.0, 3030.0, 4788.0, 6061.0, 7407.0, 9146.0]

r3=[164.0, 404.0, 482.0, 787.0, 1169.0, 1532.0]
c3=[532.0, 1040.0, 1545.0, 2076.0, 2456.0, 3022.0]
s3=[0.37450197061021706, 0.38278539610908957, 0.35129686590739045, 0.3816591349171552, 0.39697099419840176, 0.40150666470408325]

u3=[1064.0, 2081.0, 3091.0, 4152.0, 4912.0, 6045.0]

r1=[725.0, 1579.0, 2300.0, 3428.0, 4228.0, 5166.0]
c1=[140.0, 136.0, 134.0, 134.0, 136.0, 152.0]
s1=[0.4627502513670604, 0.5023413789267951, 0.4357895502535539, 0.5075283689975332, 0.5103062727112481, 0.5067371815833288]

u1=[20.0, 12.0, 9.0, 9.0, 12.0, 45.0]

r5=[227.0, 724.0, 830.0, 1464.0, 1932.0, 2527.0]
c5=[451.0, 793.0, 1186.0, 1508.0, 1720.0, 2076.0]
s5=[0.36749057709575605, 0.401842700438032, 0.349225482860864, 0.3961398173456347, 0.39992270374843025, 0.4057382648236011]

u5=[902.0, 1586.0, 2372.0, 3016.0, 3441.0, 4152.0]

r6=[133.0, 538.0, 517.0, 1042.0, 1556.0, 1989.0]
c6=[555.0, 1014.0, 1488.0, 1769.0, 1963.0, 2500.0]
s6=[0.37013060333471337, 0.41120980513667976, 0.3473740543686092, 0.3747692832555282, 0.3853787461584529, 0.3957141056569116]
u6=[1110.0, 2028.0, 2977.0, 3538.0, 3926.0, 5000.0]


x_axis = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]



# plt.plot(x_axis,r1,markersize=20, linewidth=5,marker='X', linestyle='--', color='g', label='RL')
#
# plt.plot(x_axis,r2,markersize=20, linewidth=5,marker='o', linestyle='--', color='r', label='GFC')
#
# plt.plot(x_axis,r3,markersize=20, linewidth=5,marker='h', linestyle='--', color='b', label='Random')
#
# plt.plot(x_axis,r5,markersize=20, linewidth=5,marker='s', linestyle='--', color='#9684FF', label='MIN')
#
# plt.plot(x_axis,r6,markersize=20, linewidth=5,marker='d', linestyle='--', color='y', label='MAX')
#
# plt.ylabel('Total Revenue (X100 Units)', fontsize=40)



# plt.plot(x_axis,s1,markersize=20, linewidth=5,marker='X', linestyle='--', color='g', label='RL')
#
# plt.plot(x_axis,s2,markersize=20, linewidth=5,marker='o', linestyle='--', color='r', label='GFC')
#
# plt.plot(x_axis,s3,markersize=20, linewidth=5,marker='h', linestyle='--', color='b', label='Random')
#
# plt.plot(x_axis,s5,markersize=20, linewidth=5,marker='s', linestyle='--', color='#9684FF', label='MIN')
#
# plt.plot(x_axis,s6,markersize=20, linewidth=5,marker='d', linestyle='--', color='y', label='MAX')
# plt.ylim((0.1,0.8))
# plt.ylabel('Service Rate [0-1]', fontsize=40)


plt.plot(x_axis,c1,markersize=20, linewidth=5,marker='X', linestyle='--', color='g', label='RL')

plt.plot(x_axis,c2,markersize=20, linewidth=5,marker='o', linestyle='--', color='r', label='GFC')

plt.plot(x_axis,c3,markersize=20, linewidth=5,marker='h', linestyle='--', color='b', label='Random')

plt.plot(x_axis,c5,markersize=20, linewidth=5,marker='s', linestyle='--', color='#9684FF', label='MIN')

plt.plot(x_axis,c6,markersize=20, linewidth=5,marker='d', linestyle='--', color='y', label='MAX')
plt.ylabel('Total Cost (X100 Units)', fontsize=40)



plt.xlabel('Density (veh/m)', fontsize=40)
plt.legend(loc='upper left')
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.legend(prop={'size': 33})
plt.grid()
plt.show()