import cv2
import matplotlib.pyplot as plt
import numpy as np


### Jing Yiran's code

# create SIFT object ((a feature detection algorithm))
sift = cv2.SIFT_create()
# create BFMatcher object
bf = cv2.BFMatcher()

####################
## Run whole process
####################

def detecting_mirrorLine(img, show_detail = False):
    """
    Main function

    If show_detail = True, plot matching details
    """
    # create mirror object
    mirror = Mirror_Symmetry_detection(img)

    # extracting and Matching a pair of symmetric features
    matchpoints = mirror.find_matchpoints()

    # get r, tehta (polar coordinates) of the midpoints of all pair of symmetric features
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)

    if show_detail: # visualize process in detail
        mirror.draw_matches(matchpoints, top = 10)
        mirror.draw_hex(points_r, points_theta)

    # find the best one with highest vote
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap= plt.cm.Spectral_r)
    plt.close()
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)

    return r, theta
    # add mirror line based on r and theta
#     mirror.draw_mirrorLine(r, theta, title)

# def test_case(filesPath):
#     files = sorted([f for f in glob.glob(filesPath)])
#     for file in files:
#         detecting_mirrorLine(file, "With Mirror Line")



#############################
## Mirror symmetry detection
#############################


class Mirror_Symmetry_detection:
    def __init__(self, img):
        self.image = img

        self.reflected_image = np.fliplr(self.image) # Flipped version of image

        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)



    def find_matchpoints(self):
        """
        Extracting and Matching a pair of symmetric features

        Matches are then sort between the features ki and the mirrored features mj
        to form a set of (pi,pj) pairs of potentially symmetric features.

        Ideally a keypoint at a certain spot on the object in original image should have a descriptor very similar to
        the descriptor on a point on the object in its mirrored version
        """
        # use BFMatcher.knnMatch() to get （k=2）matches
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        # these matches are equivalent only one need be recorded
        matchpoints = [item[0] for item in matches]

        # sort to determine the dominant symmetries
        # Distance between descriptors. The lower, the better it is.
        matchpoints = sorted(matchpoints, key = lambda x: x.distance)
        return matchpoints


    def find_points_r_theta(self, matchpoints:list):
        """
        Get r, tehta of the midpoints of all pair of symmetric features
        """
        points_r = [] # list of r for each point
        points_theta = [] # list of theta for each point
        for match in matchpoints:

            point = self.kp1[match.queryIdx]  # queryIdx is an index into one set of keypoints, (origin image)
            mirpoint = self.kp2[match.trainIdx] # trainIdx is an index into the other set of keypoints (fliped image)

            mirpoint.angle = np.deg2rad(mirpoint.angle) # Normalise orientation
            mirpoint.angle = np.pi - mirpoint.angle
            # convert angles to positive
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2*np.pi

            # pt: coordinates of the keypoints x:pt[0], y:pt[1]
            # change x, not y
            mirpoint.pt = (self.reflected_image.shape[1]-mirpoint.pt[0], mirpoint.pt[1])

            # get θij: the angle this line subtends with the x-axis.
            theta = angle_with_x_axis(point.pt, mirpoint.pt)

            # midpoit (xc,yc) are the image centred co-ordinates of the mid-point of the line joining pi and pj
            xc, yc = midpoint(point.pt, mirpoint.pt)
            r = xc*np.cos(theta) + yc*np.sin(theta)

            points_r.append(r)
            points_theta.append(theta)

        return points_r, points_theta # polar coordinates


    def draw_matches(self, matchpoints, top=10):
        """visualize the best matchs
        """
        img = cv2.drawMatches(self.image, self.kp1, self.reflected_image, self.kp2,
                               matchpoints[:top], None, flags=2)
        plt.imshow(img);
        plt.title("Top {} pairs of symmetry points".format(top))
        plt.show()

    def draw_hex(self, points_r: list, points_theta: list):
        """
        Visualize hex bins based on r and theta
        """
        # Make a 2D hexagonal binning plot of points r and theta
        image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap= plt.cm.Spectral_r)
        plt.colorbar() # add color bar
        plt.show()


    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        """Try to find the x and y coordinates of the hexbin with max count
        """
        for k, v in sorted_vote.items():
            # if mirror line is vertical, return the highest vote
            if vertical:
                return k[0], k[1]
            # otherwise, return the highest vote, whose y is not 0 or pi
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]


    def sort_hexbin_by_votes(self, image_hexbin):
        """Sort hexbins by decreasing count. (lower vote)
        """
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10,counts)) # get non-zero hexbins
        verts = image_hexbin.get_offsets() # coordinates of each hexbin
        output = {}

        for offc in range(verts.shape[0]):
            binx,biny = verts[offc][0],verts[offc][1]
            if counts[offc]:
                output[(binx,biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

    def draw_mirrorLine(self, r, theta, title:str):
        """
        Draw mirror line based on r theta polar co-ordinate
        """
        for y in range(len(self.image)):
            try:
                x = int((r-y*np.sin(theta))/np.cos(theta))
                self.image[y][x] = 255
                self.image[y][x+1] = 255
            except IndexError:
                continue

        # draw plot
        plt.imshow(self.image)
        plt.axis('off')
        plt.title(title)
        plt.show()

def angle_with_x_axis(pi, pj):  # 公式在文件里解释
    """
    calculate θij:
        the angle this line subtends with the x-axis.
    """
    # get the difference between point p1 and p2
    x, y = pi[0]-pj[0], pi[1]-pj[1]

    if x == 0:
        return np.pi/2

    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    """
    get x and y coordinates of the midpoint of pi and pj
    """
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

### End of Jin Yiran's code



### Original

class extractLine:
    def __init__(self, X) -> None:
        self.X = X
        self.r_theta = []
        for img in X:
            r, theta = detecting_mirrorLine(img)
            self.r_theta.append([r, theta])

    def get_sym(self):
        '''
        X: A numpy array of images of shape (L, x, y, c)
        returns a numpy array of shape (L,)
        '''
        def symmetry_metric(img, r, theta):
            def flip(img, r, theta):
                mat = np.array([
                    [-np.cos(2*theta), -np.sin(2*theta), 2*r*np.cos(theta)],
                    [-np.sin(2*theta), np.cos(2*theta), 2*r*np.sin(theta)]
                ])
                return cv2.warpAffine(img, mat, (img.shape[0], img.shape[1]))

            def scale(img):
                return (img - np.mean(img))/np.std(img)

            mirror = flip(img, r, theta)
            # plt.imshow(mirror)
            return np.mean(np.abs(scale(mirror) - scale(img)))

        sym = []
        for i in range(len(self.X)):
            sym.append(symmetry_metric(self.X[i], self.r_theta[i][0], self.r_theta[i][1]))
        return np.array(sym)



    def draw_all(self):
        def draw_line(img, r, theta):
                """
                Draw mirror line based on r theta polar co-ordinate
                """
                new = np.copy(img)
                for y in range(len(img)):
                    try:
                        x = int((r-y*np.sin(theta))/np.cos(theta))
                        new[y][x] = 255
                        new[y][x+1] = 255
                    except IndexError:
                        continue
                return new
        drawn = []
        for img in self.X:
            r, theta = detecting_mirrorLine(img)
            drawn.append(draw_line(img, r, theta))
        return np.array(drawn)

    def blur(self, k_size):
        '''
        k_size: Kernel size
        '''
        self.X = [cv2.GaussianBlur(img, (k_size, k_size), 0) for img in self.X]
        return self.X




def symmetry_metric(img, r, theta):
    def flip(img, r, theta):
        mat = np.array([
            [-np.cos(2*theta), -np.sin(2*theta), 2*r*np.cos(theta)],
            [-np.sin(2*theta), np.cos(2*theta), 2*r*np.sin(theta)]
        ])
        return cv2.warpAffine(img, mat, (img.shape[0], img.shape[1]))

    def scale(img):
        return (img - np.mean(img))/np.std(img)

    mirror = flip(img, r, theta)
    # plt.imshow(mirror)
    return np.mean(np.abs(scale(mirror) - scale(img)))

def get_sym(X):
    '''
    X: A numpy array of images of shape (L, x, y, c)
    returns a numpy array of shape (L,)
    '''
    sym = []
    r_theta = []
    for img in X:
        r, theta = detecting_mirrorLine(img)
        r_theta.append([r, theta])
        sym.append(symmetry_metric(img, r, theta))
    return np.array(sym)

def draw_line(img, r, theta):
        """
        Draw mirror line based on r theta polar co-ordinate
        """
        new = np.copy(img)
        for y in range(len(img)):
            try:
                x = int((r-y*np.sin(theta))/np.cos(theta))
                new[y][x] = 255
                new[y][x+1] = 255
            except IndexError:
                continue
        return new

def draw_all(X):
    drawn = []
    for img in X:
        r, theta = detecting_mirrorLine(img)
        drawn.append(draw_line(img, r, theta))
    return np.array(drawn)


