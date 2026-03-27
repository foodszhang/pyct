#!/usr/bin/env python
import sys
import itk
from itk import RTK as rtk
import os
import cv2
import numpy as np

# Defines the image type
ImageType = itk.Image[itk.F, 3]

# Defines the RTK geometry object
geometry = rtk.ThreeDCircularProjectionGeometry.New()
numberOfProjections = 360
firstAngle = 0.0
angularArc = 360.0
sid = 655.188  # source to isocenter distance
sdd = 721.4908  # source to detector distance
N = 512
TM = 1944
TN = 1536

dd = 74.8 / 1000

img_stack = np.zeros((numberOfProjections, N, N), dtype=np.float32)
for x in range(0, numberOfProjections):
    angle = firstAngle + x * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)
    if os.path.exists(f"../data/alway/{x}.tif"):
        print(f"../data/alway/{x}.tif")
        img = cv2.imread(f"../data1/{x}.tif", -1)
        print(img.dtype)
        img.astype(np.float32)
        img_stack[x] = img


# Writing the geometry to disk
xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
xmlWriter.SetFilename('test.xml')
xmlWriter.SetObject(geometry)
xmlWriter.WriteFile()


projectionsSource = itk.GetImageFromArray(img_stack)
projOrigin = [ -TN/2 * dd, -TM/2 * dd, 0 ] #input images are 3072x2560 pixels with a 0.14mm pixel size
projSpacing = [ dd, dd, 1.0 ]
projectionsSource.SetOrigin( projOrigin )
projectionsSource.SetSpacing( projSpacing )

ConstantImageSourceType = rtk.ConstantImageSource[ImageType]

# Create a stack of empty projection images

print("6666")


# Create reconstructed image
constantImageSource2 = ConstantImageSourceType.New()
sizeOutput = [N, N, N]
origin = [-(N-1)/2, -(N-1)/2, -(N-1)/2]
spacing = [1.0, 1.0, 1.0]
constantImageSource2.SetOrigin(origin)
constantImageSource2.SetSpacing(spacing)
constantImageSource2.SetSize(sizeOutput)
constantImageSource2.SetConstant(0.0)

# FDK reconstruction
print("Reconstructing...")
FDKCPUType = rtk.FDKConeBeamReconstructionFilter[ImageType]
feldkamp = FDKCPUType.New()
feldkamp.SetInput(0, constantImageSource2.GetOutput())
feldkamp.SetInput(1, projectionsSource)
feldkamp.SetGeometry(geometry)
feldkamp.GetRampFilter().SetTruncationCorrection(0.0)
feldkamp.GetRampFilter().SetHannCutFrequency(0.0)
print('settover')
array = itk.GetArrayFromImage(feldkamp.GetOutput())
import numpy as np
print('!!!!',array.shape, array.dtype)
array.tofile('test.raw')
print("Done!")



