import numpy as np
import cv2
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path, annot_path, anchor_stride, anchor_dims, iou_low, iou_high, img_dims,
                 to_fit=True, batch_size=25,
                 n_channels=1, n_classes=1, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.annot_path=annot_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.anchor_stride=anchor_stride
        self.anchor_dims=anchor_dims
        self.iou_low=iou_low
        self.iou_high=iou_high
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X = self._generate_X(list_IDs_temp)

        if self.to_fit:

            flag1, x_train_orig=self._generate_X(list_IDs_temp)
            class_output_list, valid_ind = self._generate_y_pom(list_IDs_temp, flag1)
            x_train=filtriraj(x_train_orig, valid_ind)
            return x_train, class_output_list
        else:
            return x_train

    def calc_iou(self, box1, box2):
        """

        :param box1: list of coordinates: row1, col1, row2, col2 [list]
        :param box2: list of coordinates: row1, col1, row2, col2 [list]
        :return: iou value
        """

        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        # respective area of the two boxes
        boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # overlap area
        interArea = max(xB - xA, 0) * max(yB - yA, 0)

        # IOU
        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp, image_path):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        images_list1=[]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            flag1[i], X[i,] = self._load_grayscale_image(os.path.join(image_path,'image'+ str(ID).zfill(6)+'.jpg'), self.img_dims)
            images_list1.append(X[i,])

        images_list = np.array(images_list1).astype(np.uint8)
        return flag1, images_list

    def filtriraj (self, X, valid_train):
        x_train = []
        for valid_ind in valid_train:
            x_train.append(X[valid_ind])
        x_train = np.array(x_train)

        return x_train
    def _generate_y_pom(self, list_IDs_temp, flag1):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        #y = np.empty((self.batch_size, *self.img_dim), dtype=int)
        object_annotations_list=[]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            annot_name=str(int(ID-1) + '.xml')
            root = ET.parse(os.path.join(self.annot_path, annot_name)).getroot()

            objects = []  # list of all objects in the image
            # cv2.imwrite(os.path.join(im_path_resized, im_name), image)
            for object in root.findall('object'):

                cl = object.find('class').text

                bb_xml = object.find('bndbox')
                bb = [np.int(bb_xml.find('xmin').text),  # min_col
                      np.int(bb_xml.find('xmax').text),  # max_col
                      np.int(bb_xml.find('ymin').text),  # min_row
                      np.int(bb_xml.find('ymax').text),  # max_row
                      ]

                annot = [bb[2], bb[0], bb[3], bb[1]]  # min_row, min_col, max_row, max_col

                if cl == 'car' and bb[3] - bb[2] + 1 > 25:
                    if flag1[i] == 1:
                        annot[0] = int(np.round(annot[0] * HScale))
                        annot[1] = int(np.round(annot[1] * WScale))
                        annot[2] = int(np.round(annot[2] * HScale))
                        annot[3] = int(np.round(annot[3] * WScale))
                        print(annot)
                        objects.append(annot)
                    else:
                        objects.append(annot)
                    # select positive car samples, height > 25px


            object_annotations_list.append(objects)

        output_class_list = []
        output_reg_list = []
        valid_inds = []

        num_anchors = len(self.anchor_dims)

        for img_ind, img_bboxes in (enumerate(object_annotations_list)):

            # print(img_bboxes)

            output_dims_class = (np.int(self.img_dims[0] / self.anchor_stride), np.int(self.img_dims[1] / self.anchor_stride), num_anchors + 1)
            output_dims_reg = (np.int(self.img_dims[0] / anchor_stride), np.int(self.img_dims[1] / self.anchor_stride), num_anchors * 4)

            output_class = np.zeros(output_dims_class).astype(np.int)
            output_reg = np.zeros(output_dims_reg).astype(np.int)

            # first position of an anchor center
            start_r = np.int(np.round(self.anchor_stride / 2))
            start_c = np.int(np.round(self.anchor_stride / 2))

            for output_row, center_row in enumerate(range(start_r, self.img_dims[0], self.anchor_stride)):  # iterate through rows of centers
                for output_col, center_col in enumerate(range(start_c, self.img_dims[1], self.anchor_stride)):  # iterate through columns of centers

                    for anchor_ind, anchor_dim in enumerate(self.anchor_dims):  # iterate through different anchor dimensions

                        half_anchor_dim_h = np.int(np.round(anchor_dim[0] / 2))
                        half_anchor_dim_w = np.int(np.round(anchor_dim[1] / 2))

                        for bbox in img_bboxes:  # iterate through annotated bounding boxes

                            # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                            anchor = [max(0, center_row - half_anchor_dim_h),
                                      max(0, center_col - half_anchor_dim_w),
                                      min(center_row - half_anchor_dim_h + anchor_dim[0], self.img_dims[0]),
                                      min(center_col - half_anchor_dim_w + anchor_dim[1], self.img_dims[1])]
                            # min_row, min_col, max_row, max_col

                            iou =self.calc_iou(bbox, anchor)

                            if iou >= self.iou_high:
                                # positive sample: set class, calculate deltas
                                output_class[output_row, output_col, anchor_ind] = 1

                                # --- set deltas ---
                                # current location minus correct location
                                delta_r = bbox[0] - anchor[0]
                                delta_c = bbox[1] - anchor[1]
                                delta_h = bbox[2] - bbox[0] - anchor_dim[0]
                                delta_w = bbox[3] - bbox[1] - anchor_dim[1]

                                # output_reg[output_row, output_col, anchor_ind * 4 + 0] = delta_r
                                # output_reg[output_row, output_col, anchor_ind * 4 + 1] = delta_c
                                # output_reg[output_row, output_col, anchor_ind * 4 + 2] = delta_h
                                # output_reg[output_row, output_col, anchor_ind * 4 + 3] = delta_w

                                # cv2.rectangle(imgs[img_ind], (anchor[1], anchor[0]), (anchor[3], anchor[2]), color=(255,0,0), thickness=1)
                                # cv2.rectangle(imgs[img_ind], (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0,0,0), thickness=1)
                                #
                                # cv2.imshow("slika", imgs[img_ind])
                                # cv2.waitKey(0)
                            if (iou < self.iou_high) and (iou > self.iou_low):
                                # IOU between iou_min and iou_max
                                # class - marked 2, deltas - 0
                                output_class[output_row, output_col, anchor_ind] = 2

            # assign background
            for out_row in range(output_class.shape[0]):  # iterate through rows of output
                for out_col in range(output_class.shape[1]):  # iterate through columns of output

                    if sum(output_class[out_row, out_col, :]) == 0:
                        # print(out_row, out_col)
                        output_class[out_row, out_col, 4] = 1

            # replace 2s with 0s
            output_class = np.where(output_class == 2, 0, output_class)

            # remove border pixels
            for ind_a, a_dim in enumerate(self.anchor_dims):
                border_padding = np.int((anchor_dims[ind_a][0] / anchor_stride) / 2) + 1

                output_class[0:border_padding, :, ind_a] = 0
                output_class[output_class.shape[0] - border_padding:, :, ind_a] = 0
                output_class[:, 0:border_padding, ind_a] = 0
                output_class[:, output_class.shape[1] - border_padding:, ind_a] = 0

                # output_reg[0:border_padding, :, ind_a] = 0
                # output_reg[output_class.shape[0] - border_padding:, :, ind_a] = 0
                # output_reg[:, 0:border_padding, ind_a] = 0
                # output_reg[:, output_class.shape[1] - border_padding:, ind_a] = 0

            # --- select negative samples ---

            # count positives and negatives
            num_positives = np.sum(output_class[:, :, 0:4])

            # find negatives
            negs = output_class[:, :, 4]
            [r, c] = np.where(negs == 1)

            # select negatives to remove
            ind_to_remove = np.arange(len(r))
            np.random.shuffle(ind_to_remove)

            num_neg = min(len(r), num_positives * 10)  # number of positive to negative samples ratio: 1 to 3
            num_to_remove = len(r) - num_neg
            ind_to_remove = ind_to_remove[:num_to_remove]

            # remove negatives
            for ind in ind_to_remove:
                output_class[r[ind], c[ind], :] = 0

            if num_positives > 0:
                output_class_list.append(output_class)
                output_reg_list.append(output_reg)
                valid_inds.append(img_ind)

        output_class_list = np.array(output_class_list)
        # output_reg_list = np.array(output_reg_list)

        output_reg_list = output_reg_list / 100.0

        return output_class_list,  valid_inds
            # Store sample


    def _load_grayscale_image(self, image_path,  im_size):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
    #    img = cv2.imread(image_path)
    #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    img = img / 255

      #  if not image_path[-4:] != '.bmp':  # exclude system files
       #     continue
        flag1=0
        image = cv2.imread(image_path, 0)
        rows, col = image.shape[:2]
        if im_size != (col, rows):
            image = cv2.resize(image, self.img_dims, interpolation=cv2.INTER_AREA)
            flag1=1
        image = image.reshape(image.shape[0], image.shape[1], 1)
        return flag1, image
    #
    # def _generate_X(self, list_IDs_temp, valid_train):
    #     """Generates data containing batch_size images
    #     :param list_IDs_temp: list of label ids to load
    #     :return: batch of images
    #     """
    #     # Initialization
    #     #X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #
    #     # Generate data
    #
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         flag1, X[i,] = self._load_grayscale_image(self.image_path +  'image'+ID)
    #     x_train = []
    #     for valid_ind in valid_train                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              :
    #         x_train.append(X[valid_ind])
    #     x_train = np.array(x_train)
    #
    #     images_list = np.array(x_train).astype(np.uint8)
    #
    #     return images_list
