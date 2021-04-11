# 数据组织
import numpy as np
X89 = np.load("/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/X89.npy")
# img_rows, img_cols, num_chan = 17, 19, 5
img_rows, img_cols, num_chan = 8, 9, 5
falx = X89
# 扩充到17*19
# if img_rows == 17:
#     tmp_z = np.zeros((1, 1, 9, 5))
#     for i in range(9):
#         idx = 8 - i
#         X89 = np.insert(X89, idx, values=tmp_z, axis=1)
#     tmp_x = np.zeros((45*3394, 17, 19, 5))
#     for i in range(9):
#         tmp_x[:, :, i*2+1, :] = X89[:, :, i, :]
#     falx = tmp_x

falx = falx.reshape((45, 6788, img_rows, img_cols, 5))

t = 6
if t == 2:   ns = 3394  # 10182
elif t == 3: ns = 2257  # 6771
elif t == 4: ns = 1692  # 5076
elif t == 5: ns = 1354  # 4062
elif t == 6: ns = 1126  # 3378
elif t == 7: ns = 962   # 2886
elif t == 8: ns = 842   # 2526
elif t == 9: ns = 746   # 2238
elif t == 10: ns = 675  # 2025
elif t == 11: ns = 611  # 1833
elif t == 12: ns = 559  # 1677

new_x = np.zeros((45, ns, t, img_rows, img_cols, 5))
new_y = np.array([])
# label = [ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
# [235, 233, 206, 238, 185,  195,  237,  216,  265,  237,  235,  233,  235,  238,  206]
# [235, 468, 674, 912, 1097, 1292, 1529, 1745, 2010, 2247, 2482, 2715, 2950, 3188, 3394]

# [470, 466, 412,  476,  370,  390,  474,  432,  530,  474,  470,  466,  470,  476,  412]
# [470, 936, 1348, 1824, 2194, 2584, 3058, 3490, 4020, 4494, 4964, 5430, 5900, 6376, 6788]

for nb in range(45):
    z = 0
    i = 0
    while i+t <= 470:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 1)
        i = i + t
        z = z + 1
    i = 470
    while i+t <= 936:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 0)
        i = i + t
        z = z + 1
    i = 936
    while i+t <= 1348:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, -1)
        i = i + t
        z = z + 1
    i = 1348
    while i+t <= 1824:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, -1)
        i = i + t
        z = z + 1
    i = 1824
    while i+t <= 2194:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 0)
        i = i + t
        z = z + 1
    i = 2194
    while i+t <= 2584:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 1)
        i = i + t
        z = z + 1
    i = 2584
    while i+t <= 3058:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, -1)
        i = i + t
        z = z + 1
    i = 3058
    while i+t <= 3490:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 0)
        i = i + t
        z = z + 1
    i = 3490
    while i+t <= 4020:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 1)
        i = i + t
        z = z + 1
    i = 4020
    while i+t <= 4494:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 1)
        i = i + t
        z = z + 1
    i = 4494
    while i+t <= 4964:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 0)
        i = i + t
        z = z + 1
    i = 4964
    while i+t <= 5430:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, -1)
        i = i + t
        z = z + 1
    i = 5430
    while i+t <= 5900:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 0)
        i = i + t
        z = z + 1
    i = 5900
    while i+t <= 6376:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, 1)
        i = i + t
        z = z + 1
    i = 6376
    while i+t <= 6788:
        new_x[nb, z] = falx[nb, i:i + t]
        new_y = np.append(new_y, -1)
        i = i + t
        z = z + 1
    print('{}-{}'.format(nb, z))


np.save('/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/t'+str(t)+'x_89.npy', new_x)
np.save('/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/t'+str(t)+'y_89.npy', new_y)