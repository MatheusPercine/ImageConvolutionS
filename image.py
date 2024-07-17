import numpy as np
import cv2
import array


class Image:

    def __init__(self, filepath: str):
        """
        Initialize the image
        :param filepath: The path to the image
        """
        self.data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # Read the image in the original format
        if self.data is None:
            raise FileNotFoundError(f"File {filepath} not found")

        self.height = self.data.shape[0]
        self.width = self.data.shape[1]

        if len(self.data.shape) == 2:
            self.channels = 1
            self.data = self.data[:, :, np.newaxis]
        else:
            self.channels = self.data.shape[2]

        self.filepath = filepath
        self.name = filepath.split('/')[-1].split('.')[0]
        self.pitch = self.width  # Can be used to optimize the padding

    @classmethod
    def from_data(cls, data: np.ndarray):
        """
        Initialize the image from data
        :param data: The image data
        :return: The image
        """
        image = cls.__new__(cls)
        image.data = data
        image.height, image.width, image.channels = data.shape
        image.filepath = None
        image.name = "Image"
        image.pitch = image.width
        return image

    def save_image(self, output_path: str):
        """
        Save the image to a file
        :param output_path: The path to save the image
        """
        cv2.imwrite(output_path, self.data)

    def show(self):
        """
        Display the image
        """
        cv2.imshow(self.name, self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # img = Image('./images/dog.jpg')
    # img.save_image_bin('./bin/dog')
    # img = Image.read_image_bin('./bin/dog', 1)
    #img.show()
    print()
