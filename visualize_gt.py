import cv2
import time
from os.path import join
import ntpath

def visualize(video_file, gt_tracks_path):
    def get_fps(start_time, frame_count):
        elapsed_time = time.time() - start_time
        return frame_count / elapsed_time if elapsed_time > 0 else 0

    gt_tracks = {}
    color_for_obj_type = {
        'person': (0, 0, 255),   # Red
        'car': (0, 255, 0),      # Green
        'vehicles': (255, 0, 0), # Blue
        'object': (255, 255, 0), # Yellow
        'bicycles': (255, 0, 255) # Magenta
    }


    obj_type_itos = {1: 'person', 2: 'car', 3: 'vehicles', 4: 'object', 5: 'bicycles'}

    with open(gt_tracks_path, 'r') as f:
        for line in f:
            obj_id, obj_duration, frame_id, x, y, w, h, obj_type = map(int, line.strip().split())
            if frame_id not in gt_tracks:
                gt_tracks[frame_id] = {}
            gt_tracks[frame_id][obj_id] = {'bbox': (x, y, w, h), 'obj_type': obj_type_itos[obj_type]}

    cap = cv2.VideoCapture(video_file)

    out_fn = video_file.replace('videos', 'gtvis')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(out_fn, fourcc, fps, frame_size)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        gt_tracks_for_frame = gt_tracks.get(frame_id, {})

        cv2.putText(frame, f'frame:{frame_id}', (frame_size[0] // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for obj_id, info in gt_tracks_for_frame.items():
            bbox = info['bbox']
            obj_type = info['obj_type']
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_for_obj_type[obj_type], 2)
            cv2.putText(frame, f'{obj_type}:{obj_id}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_for_obj_type[obj_type], 2)

        frame_count += 1
        fps = get_fps(start_time, frame_count)
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f'{ntpath.basename(video_file)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


video = 'VIRAT_S_000201_02_000590_000623'
import glob
from tqdm import tqdm
for video_file in tqdm(sorted(glob.glob(f'assets/customdata/virat/videos/*.mp4'))):
    video = video_file.split('/')[-1].split('.')[0]
    visualize(
        video_file=video_file
        ,gt_tracks_path=f'assets/customdata/virat/annotations/{video}.viratdata.objects.txt'
    )