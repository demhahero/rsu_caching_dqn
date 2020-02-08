import matplotlib.pyplot as plt


r2=[-2104.0, -1811.0, -94.0, 1058.0, 2107.0, 2501.0]
c2=[2672.0, 2596.0, 2277.0, 1732.0, 1424.0, 1321.0]
s2=[0.09904300884153885, 0.1359034726309594, 0.37918121341601096, 0.5000448017433254, 0.6470523585527762, 0.691891118982503]

u2=[5345.0, 5193.0, 4555.0, 3464.0, 2849.0, 2642.0]

r3=[-1167.0, -829.0, 615.0, 1642.0, 2618.0, 2910.0]
c3=[1785.0, 1723.0, 1546.0, 1180.0, 917.0, 889.0]
s3=[0.10777302651067641, 0.1548350240625974, 0.37562746426276206, 0.5058260187020397, 0.6476311894608813, 0.6877553199769046]

u3=[3571.0, 3446.0, 3093.0, 2360.0, 1834.0, 1779.0]

r1=[1786.0, 2173.0, 2781.0, 3007.0, 3397.0, 3599.0]
c1=[118.0, 123.0, 140.0, 109.0, 133.0, 159.0]
s1=[0.30022468033498176, 0.38322577294602364, 0.5060081982873917, 0.5264640313683886, 0.6596793716387509, 0.6992433750291859]

u1=[170.0, 260.0, 90.0, 180.0, 260.0, 380.0]

r5=[-635.0, -150.0, 1086.0, 1149.0, 2131.0, 2501.0]
c5=[1198.0, 1295.0, 1176.0, 1645.0, 1400.0, 1321.0]
s5=[0.09811598900122324, 0.19835370287020046, 0.39305750959651226, 0.5008942427967757, 0.6469992379947539, 0.691891118982503]

u5=[2396.0, 2591.0, 2353.0, 3291.0, 2800.0, 2642.0]

r6=[957.0, 828.0, 805.0, 1597.0, 2236.0, 2569.0]
c6=[72.0, 235.0, 1408.0, 1147.0, 1274.0, 1258.0]
s6=[0.17952819569179726, 0.1842329397915729, 0.38447883556528234, 0.4917654395767848, 0.6429987104526604, 0.6927599082703311]
u6=[145.0, 470.0, 2816.0, 2294.0, 2548.0, 2516.0]

x_axis = [600, 750, 1000, 1400, 2000, 3000]





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


# plt.plot(x_axis,c1,markersize=20, linewidth=5,marker='X', linestyle='--', color='g', label='RL')
#
# plt.plot(x_axis,c2,markersize=20, linewidth=5,marker='o', linestyle='--', color='r', label='GFC')
#
# plt.plot(x_axis,c3,markersize=20, linewidth=5,marker='h', linestyle='--', color='b', label='Random')
#
# plt.plot(x_axis,c5,markersize=20, linewidth=5,marker='s', linestyle='--', color='#9684FF', label='MIN')
#
# plt.plot(x_axis,c6,markersize=20, linewidth=5,marker='d', linestyle='--', color='y', label='MAX')
#
# plt.ylabel('Total Cost (X100 Units)', fontsize=40)



plt.xlabel('Cache Size in Mb', fontsize=40)
plt.legend(loc='upper left')
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.legend(prop={'size': 33})
plt.grid()
plt.show()