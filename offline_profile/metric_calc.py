n = 32
k = 2
## 512
# te = 1.432
# tp = 2.019
# td = 1.221
te = 2.890
tp = 3.170
td = 2.350

def calc_frame_duration():
    staleness = []
    frame_duration = []
    pipline_est_frame_duration = []
    for k in [2, 4, 8, 16, 32]:
        staleness.append(te + (n - k)/n * tp + (n - 2*k)/n *td)
        frame_duration.append(k * te + k * tp + (n - k + 1) * td)
        pipline_est_frame_duration.append(te + k * tp + (n - k + 1) * td)
    print(staleness)
    print(frame_duration)
    print(pipline_est_frame_duration)

calc_frame_duration()