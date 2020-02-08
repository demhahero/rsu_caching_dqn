import matplotlib.pyplot as plt


r2=[2256.0, 816.0, 337.0, 17.0, 128.0, -26.0]
c2=[1416.0, 2005.0, 2129.0, 2164.0, 2114.0, 2181.0]
s2=[0.6503956078586894, 0.4852175916849813, 0.42994534452809346, 0.39112333029247776, 0.4063262500249072, 0.3894273605450783]

u2=[2833.0, 4011.0, 4259.0, 4329.0, 4229.0, 4362.0]

r3=[2766.0, 1491.0, 1042.0, 736.0, 923.0, 714.0]
c3=[935.0, 1325.0, 1430.0, 1474.0, 1358.0, 1465.0]
s3=[0.655492743795888, 0.4843749462598581, 0.43089140513521, 0.39623483530092823, 0.4134940920099484, 0.3939889574654582]

u3=[1871.0, 2650.0, 2861.0, 2948.0, 2717.0, 2930.0]

r1=[4256.0, 3248.0, 3016.0, 2708.0, 2713.0, 2668.0]
c1=[102.0, 109.0, 115.0, 126.0, 132.0, 159.0]
s1=[0.7535890812585092, 0.5718689187655416, 0.5387591099712697, 0.4825062146366117, 0.5081306912973305, 0.4701214498070719]

u1=[25.0, 19.0, 10.0, 13.0, 24.0, 19.0]

r5=[3822.0, 1969.0, 1424.0, 1198.0, 1130.0, 1154.0]
c5=[10.0, 933.0, 1114.0, 1084.0, 1088.0, 1067.0]
s5=[0.6787016201420966, 0.49931384586811395, 0.4422545861289744, 0.409105002715263, 0.40186106668067506, 0.4017042733343575]

u5=[21.0, 1867.0, 2228.0, 2169.0, 2176.0, 2135.0]

r6=[2810.0, 1485.0, 1127.0, 731.0, 821.0, 802.0]
c6=[898.0, 1327.0, 1373.0, 1342.0, 1334.0, 1331.0]
s6=[0.6567320613214311, 0.48387967711203056, 0.435670492143167, 0.37165229578623443, 0.3906537282016633, 0.3856609706948122]
u6=[1796.0, 2655.0, 2747.0, 2684.0, 2669.0, 2663.0]


x_axis = [5, 10, 20, 30, 40, 50]



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
#
# plt.ylabel('Service Rate (%)', fontsize=40)

#
# plt.plot(x_axis,c1,markersize=20, linewidth=5,marker='X', linestyle='--', color='g', label='RL')
#
# plt.plot(x_axis,c2,markersize=20, linewidth=5,marker='o', linestyle='--', color='r', label='GFC')
#
# plt.plot(x_axis,c3,markersize=20, linewidth=5,marker='h', linestyle='--', color='b', label='Random')
#
# plt.plot(x_axis,c5,markersize=20, linewidth=5,marker='s', linestyle='--', color='#9684FF', label='MIN')
#
# plt.plot(x_axis,c6,markersize=20, linewidth=5,marker='d', linestyle='--', color='y', label='MAX')
# plt.ylabel('Total Cost (X100 Units)', fontsize=40)


plt.xlabel('Number of contents', fontsize=40)
plt.legend(loc='upper left')
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.legend(prop={'size': 33})
plt.grid()
plt.show()