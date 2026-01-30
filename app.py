import cv2
import insightface
from insightface.app import FaceAnalysis

# 1. Initialize Face Analysis (detects faces, landmarks, etc.)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load the Face Swapper model
# Make sure 'inswapper_128.onnx' is in the same directory
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)

# 3. Load your images
img1 = cv2.imread("face.jpg") # The face you WANT
img2 = cv2.imread("body.jpg") # The photo you are CHANGING

# 4. Detect faces in both images
face1 = app.get(img1)[0] # Get the first face found in source
faces_target = app.get(img2) # Get all faces in target

# 5. Perform the swap
# This loop swaps every face found in the target image with the source face
res = img2.copy()
for face in faces_target:
    res = swapper.get(res, face, face1, paste_back=True)

# 6. Save and show the result
cv2.imwrite("swapped_output.jpg", res)
cv2.imshow("Result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
