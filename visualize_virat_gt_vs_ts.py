import cv2
from os.path import join

def visualize(video_file,gt_tracks_path,pred_tracks_path,tracker,out_fn = None):
    # Read the ground truth object tracks from the file
    gt_tracks = {}
    obj_type_itos = {1:'person',2:'vehicle',3:'vehicles',4:'object',5:'cyclist'}
    obj_type_itos_pred = {0:'person',2:'vehicle'}
    with open(gt_tracks_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            obj_id = int(fields[0])
            # if obj_id != 4:
            #     continue
            obj_duration = int(fields[1])
            frame_id = int(fields[2])
            x = int(fields[3])
            y = int(fields[4])
            w = int(fields[5])
            h = int(fields[6])
            obj_type = int(fields[7])
            if obj_type>2:
                continue
            if frame_id not in gt_tracks:
                gt_tracks[frame_id] = {}
            try:
                gt_tracks[frame_id][obj_id] = {'bbox': (x, y, w, h), 'obj_type': obj_type_itos[obj_type]}
            except:
                print("Error in line: ", line)

    pred_tracks = {}
    with open(pred_tracks_path, 'r') as f:
        for line in f:
            fields = line.strip().split(' ')
            frame_id = int(fields[0])
            obj_id = int(fields[1])
            x1 = int(float(fields[2]))
            y1 = int(float(fields[3]))
            w = int(float(fields[4]))
            h = int(float(fields[5]))
            score = float(fields[6])
            obj_type = float(fields[7])
            if frame_id not in pred_tracks:
                pred_tracks[frame_id] = {}
            try:
                pred_tracks[frame_id][obj_id] = {'bbox': (x1, y1, w, h), 'obj_type': obj_type_itos_pred[obj_type], 'score': score}
            except:
                print("Error in line: ", line)

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    if out_fn is None:
        out_fn = pred_tracks_path.replace('.txt','.avi')#'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create the output video writer
    out = cv2.VideoWriter(out_fn, fourcc, fps, frame_size)



    # Get the height and width of the frames in the video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    top_center = (int(frame_width/4), 100)
    top_center3 = (int(frame_width/4), 200)
    thickness_large = 7
    thickness = 2
    color_green = (0, 255, 0) # green for ground truth tracks
    color = (0, 0, 255) # red for predicted tracks


    # Loop through the frames of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the ground truth and predicted tracks for the current frame
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        gt_tracks_for_frame = gt_tracks.get(frame_id, {})
        pred_tracks_for_frame = pred_tracks.get(frame_id, {})
        
        cv2.putText(
            frame, 'GT:frame:'+str(frame_id)
            , top_center, cv2.FONT_HERSHEY_SIMPLEX, 3.0
            , color_green, thickness_large
        )

        # Draw the ground truth tracks on the frame
        for obj_id, info in gt_tracks_for_frame.items():
            bbox = info['bbox']
            obj_type = info['obj_type']
           
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_green, thickness)
            cv2.putText(frame, f'{obj_type}:{obj_id}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, thickness)
            #cv2.putText(frame, f'Tracker: {tracker}', top_center3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness)
            cv2.putText(frame, f'Tracker: {tracker}', top_center3, cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), thickness_large)

            

        # Draw the predicted tracks on the frame
        for obj_id, info in pred_tracks_for_frame.items():
            bbox = info['bbox']
            obj_type = info['obj_type']
            
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(frame, f'{obj_type}:{obj_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            

        # Write & Display the frame
        out.write(frame)
        cv2.imshow('frame', frame)

        # Check for key press
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video = 'VIRAT_S_000201_02_000590_000623'
    visualize(
        video_file=f'assets/customdata/virat/videos/{video}.mp4'
        ,gt_tracks_path=f'assets/customdata/virat/annotations/{video}.viratdata.objects.txt'
        ,pred_tracks_path=f'/mnt/data/yolov8_tracking/runs/virat/bytetrack/{video}.txt'
    )