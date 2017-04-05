# Lane Tracker



import numpy as np
import cv2

class LaneTracker():
    
    def __init__(self,
                 H,
                 rows, 
                 cols,
                 nwindows = 8,
                 win_margin = 100,
                 min_win_pix = 50):
                     
        self.H = H
        # computing inverse homohgraphy (to transform the quadratics in the original)
        self.Hinv = np.array(np.linalg.inv(np.mat(self.H)))
        self.rows = rows
        self.cols = cols
        self.nwindows = nwindows
        self.win_margin = win_margin
        self.min_win_pix = min_win_pix
        self.first_tracked = False
        # visualization images
        self.out_img = None
        # Quadratic coefficients  of the fitted curves       
        self.left_fit = None
        self.right_fit = None
        # will fill the following tupples with 
        # end-point coordinates of the fitted parabolas        
        self.ll = None
        self.ul = None
        self.lr = None  
        self.ur = None
        
    
    def process_first_warped(self, binary_warped, image):
        
        # Take a histogram of the bottom 0.6 of the image
        histogram = np.sum(binary_warped[(int)(0.6*binary_warped.shape[0]):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        left_max_x = np.argmax(histogram[:midpoint])        
        right_max_x = np.argmax(histogram[midpoint:]) + midpoint        
        # using priors in case the histogram didnt find anythging        
        leftx_base_prior = ( (int)(self.cols/4) - 30 ) - 50
        rightx_base_prior = ( (int)(self.cols/4) - 30 + 840 ) -160         
        alpha = 1.0        
        if (histogram[left_max_x] < alpha * self.min_win_pix) and (histogram[right_max_x] < 1.5 * self.min_win_pix):
            # just locations that feel right            
            leftx_base = leftx_base_prior
            rightx_base = rightx_base_prior
        elif (histogram[left_max_x] < alpha * self.min_win_pix):
            # working-out the left from the right...
            rightx_base = right_max_x
            leftx_base = rightx_base - ( rightx_base_prior - leftx_base_prior )
        elif (histogram[right_max_x] < alpha * self.min_win_pix):
            # working out the right from the left (more possible scenario)            
            leftx_base = left_max_x
            rightx_base = leftx_base + ( rightx_base_prior - leftx_base_prior )
        else:
            leftx_base = left_max_x
            rightx_base =  right_max_x

        print("Right base : ", rightx_base)
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.win_margin
            win_xleft_high = leftx_current + self.win_margin
            win_xright_low = rightx_current - self.win_margin
            win_xright_high = rightx_current + self.win_margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            print("shape of good left indices : ", good_left_inds.shape)
            
            # If you found > self.min_win_pix pixels, recenter next window on their mean position
            if (len(good_left_inds) > self.min_win_pix) and (len(good_right_inds) > self.min_win_pix):
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            elif len(good_left_inds) > self.min_win_pix:
                # use the prior to "guess" the right base
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                rightx_current = leftx_current + ( rightx_base_prior - leftx_base_prior) + window * 2
            elif len(good_right_inds) > self.min_win_pix:
                #use the prior to "guess" the left side
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                leftx_current = rightx_current - ( rightx_base_prior - leftx_base_prior) - window * 2
            else:
                leftx_current = leftx_base_prior - window * 2
                rightx_current = rightx_base_prior + window * 2
            
            # Append these indices to the lists
            if (len(good_left_inds) > 0) and (len(good_right_inds)>0):
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
            elif len(good_left_inds) > 0:
                left_lane_inds.append(good_left_inds)
                right_guess = np.zeros(21, np.int32)
                for j in range(-10, 10):                
                    right_guess[j+10] = j + rightx_current
                right_lane_inds.append(right_guess)
            elif len(good_right_inds)>0:
                right_lane_inds.append(good_right_inds)
                left_guess = np.zeros(21, np.int32)                
                for j in range(-10, 10):                
                    left_guess[j+10] = j+leftx_current
                left_lane_inds.append(left_guess)
            else:
                left_guess = np.zeros(21, np.int32)                
                right_guess = np.zeros(21, np.int32)                
                for j in range(-10, 10):                
                    left_guess[j+10] = j + leftx_current
                    right_guess[j+10] = j + rightx_current
                left_lane_inds.append(left_guess)
                right_lane_inds.append(right_guess)
        #print("good left indices now: ", left_lane_inds)
        # just draw the prior positions of the lines
        cv2.circle(self.out_img, (rightx_base_prior, self.rows - 20), 5, color = (0, 0, 255), thickness = 3)
        cv2.circle(self.out_img, (leftx_base_prior, self.rows - 20), 5, color = (0, 0, 255), thickness = 3)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        #print("right lane indices : ", right_lane_inds)
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        ####### Display polynomials and color the lanes #######
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        A2l = self.left_fit[0]
        A1l = self.left_fit[1]        
        A0l = self.left_fit[2]
        
        left_fitx = A2l * ploty**2 + A1l * ploty + A0l
        
        A2r = self.right_fit[0]
        A1r = self.right_fit[1]        
        A0r = self.right_fit[2]
        
        right_fitx = A2r * ploty**2 + A1r * ploty + A0r

        self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]    
        
        # use opencv to draw the quadratics
        left_pts = np.zeros((len(ploty), 2), np.int32)
        right_pts = np.zeros((len(ploty), 2), np.int32)
        #print("right_fitx :", right_fitx[100])
        for i in range(len(right_fitx)):
            left_pts[i, 0] = left_fitx[i]
            right_pts[i, 0] = right_fitx[i]
            left_pts[i, 1] = ploty[i]
            right_pts[i, 1] = ploty[i]
        left_pts = left_pts.reshape((-1,1,2))
        right_pts = right_pts.reshape((-1,1,2))
                
        self.out_img = cv2.polylines(self.out_img,[left_pts],False,(0,255,255))
        self.out_img = cv2.polylines(self.out_img,[right_pts],False,(0,255,255))
        
        
        # now need to work-out end-points of the two parabolas
        # in un-warped image coordinates in order to pass it on to
        # the frame handler so that it crips these strips properly
        ul_x = left_fitx[0]
        ul_y = ploty[0]
                
        ll_x = left_fitx[len(ploty)-1]
        ll_y = ploty[len(ploty)-1]
        
        ur_x = right_fitx[0]
        ur_y = ploty[0]
        
        lr_x = right_fitx[len(ploty)-1]
        lr_y = ploty[len(ploty)-1]
        #print("Lower-left point: (", ll_x, " , ", ll_y, ")")        
        #print("Upper-left point: (", ul_x, " , ", ul_y, ")")        
        #print("Lower-right point: (", lr_x, " , ", lr_y, ")")        
        
        #print("Uppereright point: (", ur_x, " , ", ur_y, ")")        
                 
        # 1. transform the coordinates of ll
        ll_x1 = self.Hinv[0, 0] * ll_x + self.Hinv[0, 1] * ll_y + self.Hinv[0, 2]
        ll_y1 = self.Hinv[1, 0] * ll_x + self.Hinv[1, 1] * ll_y + self.Hinv[1, 2]
        ll_scale = self.Hinv[2, 0] * ll_x + self.Hinv[2, 1] * ll_y + self.Hinv[2, 2]
        # scale the coordinates and store
        ll_x1 /= ll_scale
        ll_y1 /= ll_scale
        self.ll = (ll_x1, ll_y1)
        
        # 2. transform the coordinates of ul
        ul_x1 = self.Hinv[0, 0] * ul_x + self.Hinv[0, 1] * ul_y + self.Hinv[0, 2]
        ul_y1 = self.Hinv[1, 0] * ul_x + self.Hinv[1, 1] * ul_y + self.Hinv[1, 2]
        ul_scale = self.Hinv[2, 0] * ul_x + self.Hinv[2, 1] * ul_y + self.Hinv[2, 2]
        # scale the coordinates and store
        ul_x1 /= ul_scale
        ul_y1 /= ul_scale
        self.ul = (ul_x1, ul_y1)
        
        # 3. transform the coordinates of ll
        lr_x1 = self.Hinv[0, 0] * lr_x + self.Hinv[0, 1] * lr_y + self.Hinv[0, 2]
        lr_y1 = self.Hinv[1, 0] * lr_x + self.Hinv[1, 1] * lr_y + self.Hinv[1, 2]
        lr_scale = self.Hinv[2, 0] * lr_x + self.Hinv[2, 1] * lr_y + self.Hinv[2, 2]
        # scale the coordinates and store
        lr_x1 /= lr_scale
        lr_y1 /= lr_scale
        self.lr = (lr_x1, lr_y1)
        
        
        # 4. transform the coordinates of ul
        ur_x1 = self.Hinv[0, 0] * ur_x + self.Hinv[0, 1] * ur_y + self.Hinv[0, 2]
        ur_y1 = self.Hinv[1, 0] * ur_x + self.Hinv[1, 1] * ur_y + self.Hinv[1, 2]
        ur_scale = self.Hinv[2, 0] * ur_x + self.Hinv[2, 1] * ur_y + self.Hinv[2, 2]
        # scale the coordinates and store
        ur_x1 /= ur_scale
        ur_y1 /= ur_scale
        self.ur = (ur_x1, ur_y1)
        #print("Transformed Lower-left point: (", self.ll)        
        #print("Transformed Upper-left point: (", self.ul)        
        #print("Transformed Lower-right point: (", self.lr)        
        
        #print("Transformed Uppereright point: (", self.ur)                
        
        ########## Drawing in the original image ################
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly() (!)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Hinv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        self.final_image = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        
    def process_next_warped(self, binary_warped, image):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # We now go and fetch our ready-made polynomial coefficients
        # left quadratic        
        A2l = self.left_fit[0]
        A1l = self.left_fit[1]        
        A0l = self.left_fit[2]
        # right quadratic        
        A2r = self.right_fit[0]
        A1r = self.right_fit[1]        
        A0r = self.right_fit[2]
            
        
        left_lane_inds = ((nonzerox > (A2l * (nonzeroy**2) + A1l * nonzeroy + A0l - self.win_margin)) & (nonzerox < (A2l * (nonzeroy**2) + A1l * nonzeroy + A0l + self.win_margin))) 
        right_lane_inds = ((nonzerox > (A2r * (nonzeroy**2) + A1r * nonzeroy + A0r - self.win_margin)) & (nonzerox < (A2r * (nonzeroy**2) + A1r * nonzeroy + A0r + self.win_margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # update quadratic coefficients
        A2l = self.left_fit[0]
        A1l = self.left_fit[1]        
        A0l = self.left_fit[2]
        # right quadratic        
        A2r = self.right_fit[0]
        A1r = self.right_fit[1]        
        A0r = self.right_fit[2]        
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = A2l * ploty**2 + A1l * ploty + A0l
        right_fitx = A2r * ploty**2 + A1r * ploty + A0r
        
        ######## Again, visualization on out_img *****************
        # Create an image to draw on and an image to show the selection window
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(self.out_img)
        # Color in left and right line pixels
        self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.win_margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.win_margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.win_margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.win_margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        self.out_img = cv2.addWeighted(self.out_img, 1, window_img, 0.3, 0)
        
         # use opencv to draw the quadratics
        left_pts = np.zeros((len(ploty), 2), np.int32)
        right_pts = np.zeros((len(ploty), 2), np.int32)
        #print("right_fitx :", right_fitx[100])
        for i in range(len(right_fitx)):
            left_pts[i, 0] = left_fitx[i]
            right_pts[i, 0] = right_fitx[i]
            left_pts[i, 1] = ploty[i]
            right_pts[i, 1] = ploty[i]
        
        left_pts = left_pts.reshape((-1,1,2))
        right_pts = right_pts.reshape((-1,1,2))
                
        self.out_img = cv2.polylines(self.out_img,[left_pts],False,(0,255,255))
        self.out_img = cv2.polylines(self.out_img,[right_pts],False,(0,255,255))
        
        # now need to work-out end-points of the two parabolas
        # in un-warped image coordinates in order to pass it on to
        # the frame handler so that it crips these strips properly
        ul_x = left_fitx[0]
        ul_y = ploty[0]
                
        ll_x = left_fitx[len(ploty)-1]
        ll_y = ploty[len(ploty)-1]
        
        ur_x = right_fitx[0]
        ur_y = ploty[0]
        
        lr_x = right_fitx[len(ploty)-1]
        lr_y = ploty[len(ploty)-1]
        #print("Lower-left point: (", ll_x, " , ", ll_y, ")")        
        #print("Upper-left point: (", ul_x, " , ", ul_y, ")")        
        #print("Lower-right point: (", lr_x, " , ", lr_y, ")")        
        
        #print("Uppereright point: (", ur_x, " , ", ur_y, ")")        
                 
        # 1. transform the coordinates of ll
        ll_x1 = self.Hinv[0, 0] * ll_x + self.Hinv[0, 1] * ll_y + self.Hinv[0, 2]
        ll_y1 = self.Hinv[1, 0] * ll_x + self.Hinv[1, 1] * ll_y + self.Hinv[1, 2]
        ll_scale = self.Hinv[2, 0] * ll_x + self.Hinv[2, 1] * ll_y + self.Hinv[2, 2]
        # scale the coordinates and store
        ll_x1 /= ll_scale
        ll_y1 /= ll_scale
        self.ll = (ll_x1, ll_y1)
        
        # 2. transform the coordinates of ul
        ul_x1 = self.Hinv[0, 0] * ul_x + self.Hinv[0, 1] * ul_y + self.Hinv[0, 2]
        ul_y1 = self.Hinv[1, 0] * ul_x + self.Hinv[1, 1] * ul_y + self.Hinv[1, 2]
        ul_scale = self.Hinv[2, 0] * ul_x + self.Hinv[2, 1] * ul_y + self.Hinv[2, 2]
        # scale the coordinates and store
        ul_x1 /= ul_scale
        ul_y1 /= ul_scale
        self.ul = (ul_x1, ul_y1)
        
        # 3. transform the coordinates of ll
        lr_x1 = self.Hinv[0, 0] * lr_x + self.Hinv[0, 1] * lr_y + self.Hinv[0, 2]
        lr_y1 = self.Hinv[1, 0] * lr_x + self.Hinv[1, 1] * lr_y + self.Hinv[1, 2]
        lr_scale = self.Hinv[2, 0] * lr_x + self.Hinv[2, 1] * lr_y + self.Hinv[2, 2]
        # scale the coordinates and store
        lr_x1 /= lr_scale
        lr_y1 /= lr_scale
        self.lr = (lr_x1, lr_y1)
        
        
        # 4. transform the coordinates of ul
        ur_x1 = self.Hinv[0, 0] * ur_x + self.Hinv[0, 1] * ur_y + self.Hinv[0, 2]
        ur_y1 = self.Hinv[1, 0] * ur_x + self.Hinv[1, 1] * ur_y + self.Hinv[1, 2]
        ur_scale = self.Hinv[2, 0] * ur_x + self.Hinv[2, 1] * ur_y + self.Hinv[2, 2]
        # scale the coordinates and store
        ur_x1 /= ur_scale
        ur_y1 /= ur_scale
        self.ur = (ur_x1, ur_y1)
        #print("Transformed Lower-left point: (", self.ll)        
        #print("Transformed Upper-left point: (", self.ul)        
        #print("Transformed Lower-right point: (", self.lr)        
        
        #print("Transformed Uppereright point: (", self.ur)                
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Hinv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        self.final_image = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        ############ Radius of Curvature ............... ##########

        # Define y-value where we want radius of curvature
        # Evaluate at the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*A2l*y_eval + A1l)**2)**1.5) / np.absolute(2*A0l)
        right_curverad = ((1 + (2*A2r*y_eval + A1r)**2)**1.5) / np.absolute(2*A0r)
        font = cv2.FONT_HERSHEY_SIMPLEX        
        cv2.putText(self.final_image,"Left-Right radii of curvature in pixels: "+\
                                     str(left_curverad)+"m , "+str(right_curverad)+"m", (10,70), \
                                     font, 1,(255,255,255),2)      
        
        ####### Curvature in Euclidean space #####################
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 20.0/720 # meters per pixel in y dimension (a good guess)
        xm_per_pix = 3.5/self.cols # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        cv2.putText(self.final_image,"Left-Right radii of curvature in Euclidean space: "+\
                                     str(left_curverad)+"m , "+str(right_curverad)+"m",(10,110), \
                                     font, 1,(255,255,255),2)
        # AAAND the position of the vehicle WRT lane in meters
        # (using bad variable names, but just run out of good ones...)
        x_left_ = A2l * (y_eval **2) + A1l * y_eval + A0l
        x_right_ = A2r * (y_eval **2) + A1r * y_eval + A0r
        
        middle = (x_right_ + x_left_) * 0.5
        vehicle_offset_pixels = self.cols/2 - middle
        vehicle_offset_euc = vehicle_offset_pixels * xm_per_pix
        # so, presumably, the offset from the middle of the lane 
        # is the difference of the mid-column from the mid-point between the two curves at y_eval       
        cv2.putText(self.final_image,"Offset of the vehicle from the lane center: "+\
                                     str(vehicle_offset_euc)+" m", (10, 150),font, 1,(255,255,255),2)
        
        
        
        
        
    
    # track lanes (invokes either the process_first_warped() or process_second_warped() )
    def track_lanes(self, warped_binary, image):
        if (not self.first_tracked):
            self.process_first_warped(warped_binary, image)
            self.first_tracked = True
        else:
            self.process_next_warped(warped_binary, image)
            