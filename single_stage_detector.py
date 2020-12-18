import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  B, H, W, _ = grid.shape
  A, _ = anc.shape
  anchors = torch.empty((B, A, H, W, 4), device = grid.device, dtype=grid.dtype)

  for a in range(A):
      anchors[:,a,:,:,0] = grid[:,:,:,0] - anc[a,0]/2
      anchors[:,a,:,:,1] = grid[:,:,:,1] - anc[a,1]/2
      anchors[:,a,:,:,2] = grid[:,:,:,0] + anc[a,0]/2
      anchors[:,a,:,:,3] = grid[:,:,:,1] + anc[a,1]/2

  # for b in range(B):
  #     for a in range(A):
  #         width, hight  = anc[a]
  #         for h in range(H):
  #             for w in range(W):
  #                 x_c, y_c = grid[b, h, w]
  #                 x_tl = x_c - (width/2)
  #                 y_tl = y_c - (hight/2)
  #                 x_br = x_c + (width/2)
  #                 y_br = y_c + (hight/2)
  #                 anchors[b, a, h, w] = torch.tensor([x_tl, y_tl, x_br, y_br])

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  proposals = torch.empty_like(anchors)
  anc_new = torch.zeros_like(anchors)
 
  # calculate w, h
  anc_new[:, :, :, :, 2:] = anchors[:, :, :, :, 2:] - anchors[:, :, :, :, :2]
  
  # calculate x, y
  anc_new[:, :, :, :, :2] = (anchors[:, :, :, :, :2] + anchors[:, :, :, :, 2:])/2

  # add offsets according to type of the detector
  if method == 'YOLO':
      anc_new[:, :, :, :, :2] = anc_new[:, :, :, :, :2] + offsets[:,:,:,:,:2]
  else:
      anc_new[:, :, :, :, :2] = anc_new[:, :, :, :, :2] + offsets[:,:,:,:,:2] * anc_new[:, :, :, :, 2:]
  anc_new[:, :, :, :, 2:] = anc_new[:, :, :, :, 2:].clone() * torch.exp(offsets[:,:,:,:,2:])

  # convert w,h, x_c, y_c back to x_tl, y_tl, x_br, y_br
  proposals[:,:,:,:, :2] = anc_new[:,:,:,:, :2] - anc_new[:,:,:,:, 2:]/2
  proposals[:,:,:,:, 2:] = anc_new[:,:,:,:, :2] + anc_new[:,:,:,:, 2:]/2
      
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code

  # check on the paper/ not understood
  B, A, H, W, _ = proposals.shape
  proposals = proposals.reshape(B, A*H*W, 4)
  # print('proposals shape: {}'.format(proposals.shape))
  # print('bboxes shape: {}'.format(bboxes.shape))
  
  # print('proposals unsqueezed shape: {}'.format(proposals[:,:,:2].unsqueeze(2).shape))
  # print('bboxes unsqueezed shape: {}'.format(bboxes[:,:,:2].unsqueeze(1).shape))

  tl = torch.max(proposals[:,:,:2].unsqueeze(2), bboxes[:, :, :2].unsqueeze(1))
  br = torch.min(proposals[:,:,2:].unsqueeze(2), bboxes[:, :, 2:4].unsqueeze(1))

  # print('tl shape: {}'.format(tl.shape))
  # print('br shape: {}'.format(br.shape))

  intersect = torch.prod(br - tl, dim=3) * (tl < br).all(dim=3)
  # print('intersection shape: {}'.format(intersect.shape))

  a = torch.prod(bboxes[:,:, 2:4] - bboxes[:, :, :2], dim=2)
  b = torch.prod(proposals[:,:, 2:] - proposals[:,:, :2], dim=2)
  iou_mat = torch.div(intersect, a.unsqueeze(1) + b.unsqueeze(2) - intersect)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, (1,1)),
                                    nn.Dropout(p=drop_ratio),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(hidden_dim, 5*self.num_anchors + self.num_classes, (1,1)))
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    A = self.num_anchors
    C = self.num_classes
    B, _, H, W = features.shape

    predictions = self.pred_layer(features) # shape(B, 5*A + C, 7, 7)
    anchor_data = predictions[:, :5*A, :, :].view(B, A, 5, 7, 7) # shape(B, A, 5, 7, 7)
    class_scores = predictions[:, 5*A:, :,:] # shape(B, C, 7, 7)
    conf_scores  = torch.sigmoid(anchor_data[:, :, :1, :, :].clone()) # shape(B, A, 1, 7, 7)
    offsets = anchor_data[:, :, 1:, :, :].clone()   # shape(B, A, 4, 7, 7)
    offsets[:, :, :2, :, :] = torch.sigmoid(offsets[:, :, :2, :, :]) - 0.5 * torch.ones_like(offsets[:, :, :2, :, :]) # shape(B, A, 4, 7, 7)

    if pos_anchor_idx is not None and neg_anchor_idx is not None:

        offsets  = self._extract_anchor_data(offsets, pos_anchor_idx) # (M, 7, 7)
        pos = self._extract_anchor_data(conf_scores, pos_anchor_idx) # (M, 1)
        neg = self._extract_anchor_data(conf_scores, neg_anchor_idx) # (M, 1)
        conf_scores = torch.cat((pos, neg), dim=0) # (2*M, 1)
        class_scores = self._extract_class_scores(class_scores, pos_anchor_idx) # (M, C)

    else:
        conf_scores.squeeze()
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code

    # steps 1 through 3 
    features = self.feat_extractor(images)  # B x 1280 x 7 x 7
    B = features.shape[0]                  
    grid_list = GenerateGrid(B, dtype=features.dtype, device=features.device)   # B x 7 x 7 x 2
    anchors = GenerateAnchor(self.anchor_list, grid_list)  # B x A x 7 x 7 x 4
    iou_mat = IoU(anchors, bboxes)   # B x A*7*7 x N, where N is max number of objects in the batch images

    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
    activated_anc_coord, negative_anc_coord = \
    ReferenceOnActivatedAnchors(anchors, bboxes, grid_list, iou_mat, neg_thresh=0.2, method='YOLO') # (M), (M), (M), (Mx4), (M), _, _

    # steps 3 through 4
    conf_scores, offsets, class_scores = self.pred_network(features, pos_anchor_idx=activated_anc_ind, neg_anchor_idx=negative_anc_ind) # (2*M x 1), (M x 4), (M x C)
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores) # scalor
    reg_loss = BboxRegression(offsets, GT_offsets) # scalor
    anc_per_img = torch.prod(torch.tensor(anchors.shape[1:-1])) # A * 7 * 7
    cls_loss = ObjectClassification(class_scores, GT_class, B, anc_per_img, activated_anc_ind)  # scalor

    # final loss
    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss  # scalor

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code

    with torch.no_grad():

        # get scores, offsets, classes
        features = self.feat_extractor(images)  # B x 1280 x 7 x 7
        B, _, H, W = features.shape
        conf_scores, offsets, class_scores = self.pred_network(features) # (B x A x 1 x 7 x 7), (B, A, 4, 7, 7), (B x C x 7 x 7)

        # generate proposals
        grid_list = GenerateGrid(B, dtype=features.dtype, device=features.device) # (B x 7 x 7 x 2)
        anchors = GenerateAnchor(self.anchor_list, grid_list)  # (B x A x 7 x 7 x 4)
        offsets = offsets.permute(0, 1, 3, 4, 2).contiguous()  # (B x A x 7 x 7 x 4)
        proposals = GenerateProposal(anchors, offsets, 'YOLO') # (B x A x 7 x 7 x 4)
        
        _, A, _, _, _ = conf_scores.shape
        _, C, _, _, = class_scores.shape

        # reshape classes to match the other tensors (for easier prossessing)
        class_scores = class_scores.unsqueeze(1).expand(B, A, C, H, W)  # (B, A, C, 7, 7)
        class_scores = torch.argmax(class_scores, dim=2, keepdim=True)  # (B, A, 1, 7, 7)

        for b in range(B):

            # print('image num: {}/{}'.format(b+1,B))
            # threshold the scores and get indecies for selected proposals,
            # scores and classes
            idx = conf_scores[b] > thresh   # (A x 1 x 7 x 7)
            mask = idx.permute(0,2,3,1).expand(A, H, W, 4)

            # select scores, proposals and classes 
            good_scores = conf_scores[b][idx].view(-1)
            good_proposals = proposals[b][mask].view(-1, 4)
            good_classes = class_scores[b][idx].view(-1)

            # nms to get rid of duplicate proposals and scores
            left_idx = torchvision.ops.nms(good_proposals, good_scores, nms_thresh)

            # save the final proposals, scores and classes
            final_proposals.append(good_proposals[left_idx])
            final_conf_scores.append(good_scores[left_idx].unsqueeze(1))
            final_class.append(good_classes[left_idx].unsqueeze(1))

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  
  # sort the score indecies anc calculate all box areas
  keep = []
  sort_indexing = torch.argsort(scores, descending=True)
  box_areas = torch.prod(boxes[:, 2:] - boxes[:, :2], dim=1)

  # while there are elements in sorted_indexing keep looping
  while sort_indexing.shape[0] > 0:

      # select the box to keep 
      best_box_idx = sort_indexing[0]
      best_box = boxes[best_box_idx]
      keep.append(best_box_idx)

      # intersections of all boxes with the best_box and select all areas
      # for iou computation. As we are breadcasting  we do not need an inner  
      # loop to calculate all ious
      tl = torch.max(best_box[:2], boxes[sort_indexing][:, :2])
      br = torch.min(best_box[2:], boxes[sort_indexing][:, 2:])
      intersects = torch.prod(br - tl, dim=1) * (tl < br).all(dim=1)
      a = box_areas[best_box_idx]
      b = box_areas[sort_indexing]
      
      # calculate the iou and filter out the boxes
      iou = torch.div(intersects, a + b - intersects).squeeze()
      leave = torch.where(iou <= iou_threshold)
      sort_indexing = sort_indexing[leave]

  if topk:
      keep = keep[:topk]

  keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

