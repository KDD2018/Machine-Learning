def calculate_iou(box1, box2):

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    iou = area / (s1 + s2 - area)
    return iou


def prior_box(w, h, s, t, P, Q):
    prior_boxes = []
    x = 0
    y = 0
    while y + h <= Q:
        while x + w <= P:
            prior_boxes.append([x, y, x + w, y + h])
            x += s
        y += t
        x = 0
    return prior_boxes


def count_prior(prior_boxes, gt_boxs, k):
    count = 0
    for p in prior_boxes:
        for g in range(k):
            if calculate_iou(p, g):
                count += 1
                break
    return count


def get_inputs():
    w, h, s, t, k, P, Q = list(map(int, input().strip().split()))
    gt_boxes = []
    for i in range(k):
        X, Y, W, H = list(map(int, input().strip().split()))
        gt_boxes.append([X, Y, X + W, Y + H])
    return w, h, s, t, P, Q, k, gt_boxes


if __name__ == '__main__':
    w, h, s, t, P, Q, k, gt_boxes = get_inputs()
    priors = prior_box(w, h, s, t, P, Q)
    count_p = 0
    for gt in gt_boxes:
        for p in priors:
            iou = calculate_iou(gt, p)
            if iou > 0:
                count_p += 1
    print(count_p)
