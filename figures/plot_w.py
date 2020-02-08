import matplotlib.pyplot as plt

r2=[2136.0, 1231.0, 360.0, -652.0, -1313.0, -2384.0]
c2=[0.0, 952.0, 1822.0, 2694.0, 3448.0, 4443.0]
s2=[0.3722577115156077, 0.3780545649690129, 0.37918121341601096, 0.36594422362162954, 0.39107537990357705, 0.37277214371946793]

u2=[4529.0, 4760.0, 4555.0, 4490.0, 4310.0, 4443.0]

r3=[2169.0, 1657.0, 925.0, 266.0, -64.0, -911.0]
c3=[0.0, 632.0, 1237.0, 1828.0, 2235.0, 3010.0]
s3=[0.3780533280360771, 0.3963784925388637, 0.37562746426276206, 0.37549595529861257, 0.39757257367271875, 0.37996137507624533]

u3=[3027.0, 3160.0, 3093.0, 3048.0, 2794.0, 3010.0]

r1=[2739.0, 2977.0, 2982.0, 2685.0, 2764.0, 2623.0]
c1=[0.0, 114.0, 123.0, 129.0, 135.0, 143.0]
s1=[0.4822167971813021, 0.5036045424644255, 0.5060081982873917, 0.5008368965653192, 0.5165574214914787, 0.484471676564228]

u1=[27.0, 24.0, 9.0, 16.0, 24.0, 19.0]

r5=[2170.0, 1795.0, 1321.0, 661.0, 300.0, -132.0]
c5=[0.0, 474.0, 941.0, 1412.0, 1817.0, 2244.0]
s5=[0.37820841218229534, 0.39297683758612334, 0.39305750959651226, 0.3716913912554165, 0.3880163684588444, 0.3821858376486218]

u5=[2256.0, 2372.0, 2353.0, 2354.0, 2272.0, 2244.0]

r6=[2172.0, 1631.0, 1086.0, 410.0, -100.0, -679.0]
c6=[0.0, 576.0, 1126.0, 1640.0, 2242.0, 2680.0]
s6=[0.3785325206227064, 0.3822144514074023, 0.38447883556528234, 0.36764848193772914, 0.3922623496138685, 0.36218920420711215]
u6=[2789.0, 2881.0, 2816.0, 2734.0, 2802.0, 2680.0]



x_axis = [0, 0.2, 0.4, 0.6, 0.8, 1]



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
# plt.ylim((0,1))
# plt.ylabel('Service Rate (%)', fontsize=40)


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


plt.xlabel('w', fontsize=40)
plt.legend(loc='upper left')
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.legend(prop={'size': 33})
plt.grid()
plt.show()