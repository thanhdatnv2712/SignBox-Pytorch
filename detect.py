from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/RFB_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    label_mapping= ["straight", "dont turn right", "dont turn left", "stop", "turn left", "turn right"]

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing begin
    rootdir= "/home/ubuntu/seg/SignDataset/test"
    newrootdir= rootdir + "_result"
    filepaths= []

    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            path= os.path.join(root, filename)
            filepaths.append(path)

    for image_path in tqdm(filepaths): 
        # image_path = "/home/ubuntu/cds/SignDataset/images/00001.jpg"

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = args.long_side
        max_size = args.long_side
        tic = time.time()
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape


        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        #tic = time.time()
        loc, conf = net(img)  # forward pass
        #print (loc, conf)
        #exit()
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1:]
        #print (scores[0])
        #print (scores[1])
        #print (scores[2])
        
	#print (scores[3])
        #print (scores[4])
        #print (scores[5])
        #exit()
        scores = np.max(scores, axis= 1)

        # [:, 1]
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        #_pos = _pos[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]
        #_pos = _pos[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        print('net forward time: {:.4f}'.format(time.time() - tic))
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                #text = "{:.4f} {}".format(b[4], label_mapping[int(b[5])])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cx = b[0]
                cy = b[1] + 12
                #cv2.putText(img_raw, text, (cx, cy),
                #            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            # save image

            # name = "test.jpg"
            newpath= image_path.replace(rootdir, newrootdir)
            dn= os.path.dirname(newpath)

            if not os.path.exists(dn):
                os.makedirs(dn)
            cv2.imwrite(newpath, img_raw)

