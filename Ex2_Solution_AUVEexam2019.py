#!/usr/bin/env python3

# Modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Main class
class occupancy_mapping_algorithm:

    # Initialize variables
    def __init__ (self,pof,poo,pinit,meas,meas_limit,map_res,map_length):
        self.pof= pof # Prob of occupancy of a cell before a measurement
        self.poo= poo # Prob of occupancy of a cell after a measurement but less than measurement+meas_limit
        self.pinit= pinit # Initial prob of all cells, how much we know from our map
        self.meas= meas # Set of measurments
        self.meas_limit=meas_limit # End of perceptual field of the sensor
        self.map_res= map_res # Length of each cell
        self.map_length= map_length+1 # Map length
    
    # Computes new the log inverse sensor model for a given cell and measurement
    # returns l(mi|zj,xj)
    def log_inv_sensor_model (self,i,j):
        # If cell before measurement, then assign the log prob of occupancy of a cell before the measurement
        if self.cells[j]<self.meas[i]: 
            return np.log(self.pof/(1-self.pof))
        # Otherwise, assign the log prob of occupancy of a cell after the measurement
        else:
            return np.log(self.poo/(1-self.poo))
    
    # Iterates the l(mi|z1:j-1,x1:j-1)
    def occupancy_grid_mapping(self,l0,logodds):
        for i in range(len(self.meas)):
            for j in range(len(self.cells)):
                # If out of range, dont update
                if self.cells[j]>self.meas[i]+ self.meas_limit:
                    logodds[j]=logodds[j] 
                # Otherwise, update the logodds with the recursive, inverse sensor model term and prior term
                else:
                    logodds[j]=logodds[j]-l0[j] + self.log_inv_sensor_model(i,j)
        return logodds # Returns l(mi|z1:j,x1:j)
    
    # Calls the algorithm 
    def main_mapping(self):

        # Get the grid
        self.cells= range(0,self.map_length,self.map_res)
        
        # Computes the prior term based of the initial prob of all cells
        # Important: This should be log=(1-pinit/pini) because its definition its inverted in the equation
        l0= np.ones(len(self.cells))*np.log((1-self.pinit)/self.pinit)
        
        # The initial logodd value should also use the initial prob of all cells!
        logodds= np.ones(len(self.cells))*np.log(self.pinit/(1-self.pinit))
        
        # Update the logoods for the entire map
        logodds= self.occupancy_grid_mapping(l0,logodds)
        
        # Comes back to probability
        m= 1 - 1./(1+np.exp(logodds))

        # Visualizating the results
        w= 250*(len(self.cells)-1) # width of each cell in px
        h = 250 # heigth of each cell in px
        data = np.zeros((h, w)) # create matrix
        for i in range(len(self.cells)-1): # ignoring the value for 0
            data[0:h,i*250:((i+1)*250)]= np.ones([h,250])* (1-m[i+1]) *255 #assigning intensity equal to prob of being free
            data[0:h,i*250]= 127 # to visualize where the cell ends
        
        # Converting matrix to image
        img = Image.fromarray(data) 
        imgs = img.convert("L")
        imgs = imgs.save("map.jpg")
        img.show()

        # Plotting the results
        plt.plot(self.cells, m)
        plt.xlim(0, self.map_length)
        plt.ylim(0, 1.05)
        plt.xlabel("Distance [cm]") 
        plt.ylabel("Occupancy probability [Prob(mi|z1:t,x1:t)]") 
        plt.savefig("Occupancy_probability_graph.jpg")
        plt.show()

        return 0

def main():

    # Customize values of the map and sensor
    pof=0.1
    poo=0.6
    pinit=0.8
    meas=np.array([51])
    meas_limit=30
    map_res=20
    map_length=100

    # Create an instance of the class
    map = occupancy_mapping_algorithm(pof,poo,pinit,meas,meas_limit,map_res,map_length)
    map.main_mapping() # Perform the algorithm

    return 0

if __name__ == '__main__':
    main()