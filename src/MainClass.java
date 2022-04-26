import net.sourceforge.tess4j.*;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class MainClass {

	public static void readPlate() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Imgcodecs imageCodecs = new Imgcodecs();
		Mat img = Imgcodecs.imread("./1.png");

		//convert to gray-scale image and apply a bilateral filter
		Mat gray = new Mat();
		Mat gray2 = new Mat();
		Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
		Imgproc.bilateralFilter(gray, gray2, 13, 15, 15);

		//Emphasizes edges on the image
		Mat edged = new Mat();
		Imgproc.Canny(gray2, edged, 30, 200);
		HighGui.imshow("edged", edged);
		HighGui.waitKey();

		//Finds all shapes
		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchey = new Mat(); 
		Imgproc.findContours(edged.clone(), contours, hierarchey, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		//Sorts all found shapes by size, from largest to smallest
		Optional<MatOfPoint> largest = contours.stream().max(new Comparator<MatOfPoint>() {
			public int compare(MatOfPoint c1, MatOfPoint c2) {
				return (int) (Imgproc.contourArea(c1) - Imgproc.contourArea(c2));
			}
		});
		contours.sort(new Comparator<MatOfPoint>() {
			public int compare(MatOfPoint c1, MatOfPoint c2) {
				return (int) (Imgproc.contourArea(c2) - Imgproc.contourArea(c1));
			}
		});

		//Finds the largest polygon with 4 sides, and exits
		MatOfPoint2f screenCnt = new MatOfPoint2f();
		int index = 0;
		for (int i = 0; i < contours.toArray().length && i < 10; i++) {
			MatOfPoint2f c2f = new MatOfPoint2f(contours.get(i).toArray());
			double peri = Imgproc.arcLength(c2f, true);
			MatOfPoint2f approx = new MatOfPoint2f();
			Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true);

			if (approx.toArray().length == 4) {
			    Rect next = Imgproc.boundingRect(contours.get(i));
			    Rect curr = Imgproc.boundingRect(contours.get(index));
			    double nextRatio = next.width/next.height, currRatio = curr.width/curr.height;
			    if(Math.abs(nextRatio - 2.0) < Math.abs(currRatio - 2.0)) {
			        screenCnt = approx;
			        index = i;
			    }
			}
		}

		//Draws found contour onto image
		boolean found = !screenCnt.empty();
		if (found) {
			Imgproc.drawContours(img, contours, index, new Scalar(0, 0, 255), 2, Imgproc.LINE_8);
		}
		HighGui.imshow("contours", img);
		HighGui.waitKey();

		Rect rect = Imgproc.boundingRect(contours.get(index));
		img = img.submat((int) (rect.y + (rect.height * 0.35)), (int) (rect.y + (rect.height * 0.88)), (int) (rect.x + (rect.width * 0.05)), (int) (rect.x + (rect.width * 0.95)));
		HighGui.imshow("cropped", img);
		HighGui.waitKey();

		//Imgproc.resize(img, img, new Size(500, 300));
		//Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2GRAY);
		Imgproc.GaussianBlur(img, img, new Size(5, 5), 0);
		Imgcodecs.imwrite("./output.png", img);

		try {
			Tesseract it = new Tesseract();
			it.setDatapath("./tessdata");
			String str = it.doOCR(new File("./output.png"));
			System.out.println("OUTPUT 1\n" + str + "END OUTPUT 1");
		} catch (TesseractException e) {
			e.printStackTrace();
		}
	}

	public static void main(String args[]) throws Exception {
		readPlate();
	}
}
