import re
import math
from tqdm import tqdm
import os
from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
from scipy.spatial import ConvexHull
from numpy import *


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_3d_box(box_3d):

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    x, y, z, l, w, h, r =  box_3d
    
    z = z - h/2 # 将中心点从地面移到车辆中心
    
    heading_angle = r * np.pi / 180
    R = roty(heading_angle)
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + x
    corners_3d[1,:] = corners_3d[1,:] + y
    corners_3d[2,:] = corners_3d[2,:] + z
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def calculate_iou(box1, box2):
    """
    快速计算两个水平矩形框的 IoU（Intersection over Union）。

    参数:
    - box1, box2: (x1, y1, x2, y2) 矩形框，表示左上角和右下角的坐标。
    
    返回:
    - IoU (float): 交并比（Intersection over Union）
    """
    
    corners_3d_ground  = get_3d_box(box1) 
    corners_3d_predict = get_3d_box(box2)
    IOU_3d,IOU_2d =box3d_iou(corners_3d_predict,corners_3d_ground)
    
    # 计算IoU
    return IOU_3d, IOU_2d

def extract_bboxes(output):
    """
    Extract bounding box coordinates from the given string using regular expressions.
    :param output: String containing bounding box coordinates in the format {<bx_left><by_top><bx_right><by_bottom>|θ}
    :return: List of bounding boxes, each in the format [bx_left, by_top, bx_right, by_bottom, θ]
    """
    # 修改正则表达式，确保最后一个数字和管道符号能够正确匹配
    pattern = r'\[([-0-9., ]+)\]'
    matches = re.findall(pattern, output)
    bboxes = [list(map(float, match.split(","))) for match in matches]

    return bboxes

# 读取JSONL文件并将每行解析为Python字典，存入列表
def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data

def folder_creat_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def world2pixel_tup(world_xyz, incident):
    focal_length = 12 # Focal length in millimeters
    pixel_size = 17.3 / 4000 # Size of a single pixel in millimeters
    size_img = [4000, 2250]
    
    
    incident = incident - 90
    
    R_cam2drone = np.array([
        [1, 0, 0],
        [0, np.cos(incident * math.pi / 180), -np.sin(incident * math.pi / 180)],
        [0, np.sin(incident * math.pi / 180), np.cos(incident * math.pi / 180)]
    ])
    
    camera_xyz = np.dot(R_cam2drone, world_xyz)
    
    
    # Calculate the new image coordinates
    ix = camera_xyz[0] / camera_xyz[2] * focal_length 
    iy = camera_xyz[1] / camera_xyz[2] * focal_length
    image_xy = [ix, iy]
    
    # Calculate the new pixel coordinates
    nx = image_xy[0] / pixel_size + size_img[0] / 2
    ny = image_xy[1] / pixel_size + size_img[1] / 2
    
    return (nx, ny)

def draw_cube(draw, points, color):
    """
    根据给定的 8 个点绘制立方体
    :param draw: PIL.ImageDraw 对象
    :param points: 8 个点的列表，前 4 个为底面，后 4 个为顶面
    """
    # 底面四个点 (p1, p2, p3, p4)
    draw.line([points[0], points[1], points[2], points[3], points[0]], fill=color, width=5)  # 底面

    # 连接底面和顶面的点 (p1-p5, p2-p6, p3-p7, p4-p8)
    for i in range(4):
        draw.line([points[i], points[i + 4]], fill=color, width=5)  # 连接上下层
        
    # 顶面四个点 (p5, p6, p7, p8)
    draw.line([points[4], points[5], points[6], points[7], points[4]], fill=color, width=5)  # 顶面
    
def tdbox2poly(tdbox, incident):
    
    incident = int(incident)
    x, y, z, w, h, l, a = tdbox
     
    angle_rad = math.radians(a)
    cosa = math.cos(angle_rad)
    sina = math.sin(angle_rad)
        
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    
    pixel_xy_0 = world2pixel_tup([p1x, p1y, z], incident)
    pixel_xy_1 = world2pixel_tup([p2x, p2y, z], incident)
    pixel_xy_2 = world2pixel_tup([p3x, p3y, z], incident)
    pixel_xy_3 = world2pixel_tup([p4x, p4y, z], incident)
    pixel_xy_4 = world2pixel_tup([p1x, p1y, z - l], incident)
    pixel_xy_5 = world2pixel_tup([p2x, p2y, z - l], incident)
    pixel_xy_6 = world2pixel_tup([p3x, p3y, z - l], incident)
    pixel_xy_7 = world2pixel_tup([p4x, p4y, z - l], incident)

    return [pixel_xy_0, pixel_xy_1, pixel_xy_2, pixel_xy_3, pixel_xy_4, pixel_xy_5, pixel_xy_6, pixel_xy_7]

def eval_fg(answers_file, fg_flag=False, cls=""):
    # 从 jsonl 文件中加载数据
    predict = load_jsonl(answers_file)
    total_cnt = len(predict)
    correct = 0
    bev_correct = 0
    format_error = 0
    i = 0
    total_cnt_true = 0
    for i, predict in tqdm(enumerate(predict), total=total_cnt):
        
        if fg_flag:
            if cls not in predict['qtype']:
                continue

        total_cnt_true += 1

        answer = predict['answer']
        answer = answer.strip()
        gt_bbox = predict['bbox_3d']

        try:
            predict_boxes = extract_bboxes(answer)
        except:
            format_error += 1
            continue            

        ori_img_path = args.image_folder + predict['image_id']
        image = Image.open(ori_img_path)
        width, height = image.size
        vis_dir = args.vis_dir

        if vis_dir:
            folder_creat_if_not_exist(vis_dir)
            draw = ImageDraw.Draw(image)
        
        try:
            pred_bbox = predict_boxes[0]

            # compute  IoU
            iou_score, bev_score = calculate_iou(gt_bbox, pred_bbox)
            if iou_score >= 0.5:
                correct += 1

            if bev_score >= 0.5:
                bev_correct += 1

            if vis_dir:
                points = tdbox2poly(pred_bbox, predict['pitch_angle'])
                draw_cube(draw, points, "red")
                points = tdbox2poly(gt_bbox, predict['pitch_angle'])
                draw_cube(draw, points, "green")
                # draw.rectangle(gt_bbox, outline="red", width=5)
                # draw.rectangle(pred_bbox, outline="blue", width=5)
                image.save(vis_dir + predict['image_id'].split('.')[0] + f'_{i}.jpg')
        except:
            format_error += 1
            continue

    if total_cnt_true == 0:
        print(f"There are no questions related to {cls}.")
        return
    
    if fg_flag:
        print(f"{cls}:")
    else:
        print("ALL:")
    print(f'3D Precision @ 0.5: {round(correct / total_cnt_true, 5)} BEV Precision @ 0.5: {round(bev_correct / total_cnt_true, 5)} Format error ratio: {round(format_error / total_cnt_true, 5)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--image-folder', required=True, help='Image directory')
    parser.add_argument('--answers-file', required=True, help='Target jsonl directory')
    parser.add_argument('--vis-dir', default=None, help='Base URL for the API')

    args = parser.parse_args()


    eval_fg(args.answers_file, False)

    for cls in ["color", "type", "pos", "abs_size", "rel_size", "abs_dis", "rel_dis"]:
        eval_fg(args.answers_file, True, cls)
