import cv2
import os
import numpy as np
import random
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()

parser.add_argument('--bg_path', default='./background', help='Back ground pictures path', type=str)
parser.add_argument('--fg_path', default='./bird', help='Fore ground pictures path', type=str)
parser.add_argument('--obj_name', default='bird', help='The name of objects in fg_path', type=str)
parser.add_argument('--standard_size', default=608, help='A standard size to regular the objects size', type=int)
parser.add_argument('--obj_size_range', default=(15, 80), help='The size of objects range from and to', type=tuple)
parser.add_argument('--num_to_pick', default=(1, 6), help='The number of forehead pictures to pick', type=tuple)
parser.add_argument('--epoch', default=3, help='The number of epochs to run cover', type=int)
parser.add_argument('--transparent_pixel', default=255, help='Pixel value of transparent areas,to be 0 or 255',
                    type=int)

args = parser.parse_args()

IMGTEXT = ['.jpg', '.png', '.jpeg']


def random_pick(dirpath, least=args.num_to_pick[0], most=args.num_to_pick[1]):
    number = random.randint(least, most)
    foreground_list = [os.path.join(dirpath, img) for img in os.listdir(dirpath) if
                       os.path.splitext(img)[-1].lower() in IMGTEXT]
    start = random.randint(0, len(foreground_list) - 1)
    pick_list = foreground_list[start:start + number]
    return pick_list


def pretty_xml(element, indent, newline, level=0):
    '''
    方法来自https://blog.csdn.net/u012692537/article/details/101395192
    给xml文件增加缩进、换行，使其更漂亮易读
    :param element: 传入的Elment类
    :param indent: 缩进方式，一般是\t
    :param newline: 换行方式，一般是\n
    :param level:
    :return: 美化后的root
    '''
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


class ImgCover(object):
    '''
    获取一张图片作为背景，将随机选择的目标图片随机覆盖在背景的各个位置，并且创建标注文件。
    '''
    def __init__(self, background_path, pick_list, standard_size=args.standard_size):
        super(ImgCover, self).__init__()
        self.bg_path = background_path
        self.bg = cv2.imread(background_path)
        self.h_ratio = standard_size / self.bg.shape[0]
        self.w_ratio = standard_size / self.bg.shape[1]
        self.fg_list = [cv2.imread(img) for img in pick_list]
        self.img_result, self.xml = self._run_cover()

    def _resize(self, img):
        # big_thresh = min(self.bg.shape[:2]) // 8            # 控制覆盖上去的物体最大尺寸（小于背景图片尺寸//8）
        big_thresh = max(args.obj_size_range[1] / self.h_ratio,
                         args.obj_size_range[1] / self.w_ratio)  # 控制最大尺寸（resize到standard size后不大于80）
        small_thresh = min(args.obj_size_range[0] / self.h_ratio,
                           args.obj_size_range[0] / self.w_ratio)  # 控制最小尺寸（resize到standard size后不小于15）
        st_h = img.shape[0] * self.h_ratio  # 目标在standard size上面的 h
        st_w = img.shape[1] * self.w_ratio  # 目标在standard size上面的 w
        bigger, smaller = max(st_h, st_w), min(st_h, st_w)
        assert small_thresh / smaller < big_thresh / bigger, 'Please got the global param \'args.obj_size_range\' range bigger'
        resize_ratio = random.uniform(small_thresh / smaller, big_thresh / bigger)
        resized_img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        return resized_img

    def _run_cover(self):
        bg = self.bg
        root = ET.Element("annotation")
        filename = ET.SubElement(root, "filename")
        filename.text = 'cover_' + os.path.split(self.bg_path)[-1]
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, 'width')
        width.text = str(bg.shape[1])
        height = ET.SubElement(size, 'height')
        height.text = str(bg.shape[0])
        depth = ET.SubElement(size, 'depth')
        depth.text = str(bg.shape[2])

        for fg in self.fg_list:
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, 'name')
            name.text = args.obj_name
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = str(0)
            bndbox = ET.SubElement(object, 'bndbox')

            fg = self._resize(fg)
            if args.transparent_pixel == 255:    # 根据经验，透明区域转换为灰度图之后，像素值可能是0或者255。具体原因不详。
                fg[fg == 255] = 0
            # 从背景中提取出要替换的区域
            rows, cols = fg.shape[:2]
            h_start = random.randint(0, bg.shape[0] - rows)
            w_start = random.randint(0, bg.shape[1] - cols)
            roi = bg[h_start:h_start + rows, w_start:w_start + cols]
            # 位置写入xml文件
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(w_start)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(h_start)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(w_start + cols)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(h_start + rows)

            # 创建掩膜
            img2gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 0, 255,
                                      cv2.THRESH_BINARY_INV)  # >0的像素点，全部变成0，其他变成255( cv2.THRESH_BINARY_INV)
            # 保留除目标外的背景
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)  # 按位与
            dst = cv2.add(roi_bg, fg)  # 进行融合
            bg[h_start:h_start + rows, w_start:w_start + cols] = dst  # 融合后放在原图上
        pretty_xml(root, '\t', '\n')
        tree = ET.ElementTree(root)
        return bg, tree

    def write(self, output_dir, epoch):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_name = 'cover{}_'.format(epoch) + os.path.split(self.bg_path)[-1]
        output_path = os.path.join(output_dir, output_name)
        # cv2.imshow('result', self.img_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(output_path, self.img_result)
        self.xml.write(os.path.splitext(output_path)[0] + '.xml')
        print('The image {} has been covered {} objects, and rewrote in {}'.format(self.bg_path, len(self.fg_list),
                                                                                   output_path))


if __name__ == '__main__':
    for i in range(args.epoch):
        for bgpath in [os.path.join(args.bg_path, bg) for bg in os.listdir(args.bg_path) if
                       os.path.splitext(bg)[-1].lower() in IMGTEXT]:
            pick_list = random_pick(args.fg_path)
            imgcover = ImgCover(bgpath, pick_list)
            imgcover.write('./results', i)
