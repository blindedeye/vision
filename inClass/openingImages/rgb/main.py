import cv2
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    img = cv2.imread('file.jpg',cv2.IMREAD_GRAYSCALE)

    # print(f"Image size: {img.shape}")

    # img_slice = img[100:180, :]


    # img_swap = np.copy(img_slice)
    # img_swap = img[100:180:-1,:]

    # img = cv2.cvtColor(img,cv2.BGR2RGB)
    # plt.matshow(img)
    # plt.show()
    
    # x = np.linspace(len(img[0]), 0, len(img[0]))
    # y = np.linspace(0, len(img), len(img))
    # X, Y = np.meshgrid(x,y)
    # fig = plt.figure(dp1=len(img)/2)
    # ax = fig.add_subplot(111,projection='3d')
    # ax.plot_surface(X,Y,img,cmap='Greys',linewidth=1,antialiased=True,alpha=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.elev=74
    # ax.dist=12
    # ax.azim=140
    # fig.show()

    cv2.imshow('image', img)
    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    main()

