to execute : python main.py

Parameters : no of clusters : For chosen dataset best performance is for values between 3-6.
             Key frame threshold : above 100 (for extraction of key frames)
             bg pixel updatethresh : no of frames for which colour of pixel shld be const to be considered fr updation
             bg reclustering thresh : min no of pixels that have changed so dat background needs to be reclustered.
             
             
   main.py : gui
   algo.py : basic flow of project
             1)Background frame is first clustered and centers are extracted
             2)Each frame is tested if it varies a lot from prev frame, it is identified as key frame
             3) Key frame clustering using centers obtain from clustering of bg frame.
             4) Difference of both clustered images is taken.. (motion detection)
             5) This cropped area is passed to hog.
             6) From non human region from cropped area and areas rejected by hog are sent for Background Updation
                i.e. If the pixel remains constant fr given no of frames...it gets updated in bg
                If more no of pixels of bg are updated then whole bg is re-clustered to extract new centers.
                
   fuzzy_c_means.py : Fuzzy Cmeans clustering code.
   
   In results Folder : 
   C : Input frames
   C_res : Results for c Dataset.
