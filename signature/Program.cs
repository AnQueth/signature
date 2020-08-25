using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace signature
{
    class Program
    {
        static void Main(string[] args)
        {
            string root = "";

            string name = "Lien Relase 1 - Signature area contains seller1 sign contents.png";

            var b = RunOn(root + name, new Rect(550, 800, 290, 50));
           
            Console.WriteLine(name + "  lh1   "  + b);

         
              b = RunOn(root + name, new Rect(50, 800, 290, 50));
            Console.WriteLine(name + "  seller   " + b);

            name = "Seller2 sign - Signature area contains seller1 sign contents.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(120, 820, 450 - 120, 850 - 820));
            Console.WriteLine(name + "  1 owner   " + b);
            b =  RunOn(root + name, new OpenCvSharp.Rect(80, 950, 550 - 80, 990 - 950));
            Console.WriteLine(name + "  notary   " + b);


            name = "Lien Release 1 - Sign area 1 for IN format.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(60, 930, 360 ,36));
            Console.WriteLine(name + "  lr1   " + b);
            b = RunOn(root +name, new OpenCvSharp.Rect(480, 930, 360, 36));
            Console.WriteLine(name + "  lr2   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 670, 360, 36));
            Console.WriteLine(name + "  lr3   " + b);



            Cv2.WaitKey();
           
        }
        static bool RunOn(string file, OpenCvSharp.Rect boundingBox)
        {
            //  OpenCvSharp.Mat m = new OpenCvSharp.Mat(root + "Seller2 sign - Signature area contains seller1 sign contents_2.png", OpenCvSharp.ImreadModes.Grayscale);
            OpenCvSharp.Mat m = new OpenCvSharp.Mat(file, OpenCvSharp.ImreadModes.Grayscale);



            var g = m.AdaptiveThreshold(255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 3, 20);
            // g = g.Blur(new Size(5,5));

           //  Cv2.ImShow("ggg", g);

            var element = Cv2.GetStructuringElement(
                                MorphShapes.Rect,
                                new Size(10, 1));

            var h = g.MorphologyEx(MorphTypes.Close, element, iterations: 2);
          //    Cv2.ImShow("dilate", h);
            element = Cv2.GetStructuringElement(
                                MorphShapes.Rect,
                                new Size(100, 1));
            var mask = h.MorphologyEx(MorphTypes.Open, element, iterations: 2);
              // Cv2.ImShow("mask", mask);

            Mat newMask = new Mat();
            Cv2.BitwiseNot(mask, newMask);

            //  Cv2.ImShow("newMask", newMask);

            Mat newImage = new Mat();
            g.CopyTo(newImage, newMask);

            // Cv2.ImShow("newImage", newImage);

            Cv2.BitwiseNot(newImage, newImage);

            element = Cv2.GetStructuringElement(
                                 MorphShapes.Ellipse,
                                 new Size(2, 2));

            var d2 = newImage.MorphologyEx(MorphTypes.Dilate, element);
            //  Cv2.ImShow("d2", d2);
            element = Cv2.GetStructuringElement(
                             MorphShapes.Rect,
                             new Size(7, 7));
            d2 = d2.MorphologyEx(MorphTypes.Erode, element);
            //  Cv2.ImShow("d3", d2);
            // g = g.MorphologyEx(MorphTypes.Close, element, iterations: 4);
            // g = g.GaussianBlur(new Size(5, 5), 5);

            // Rect r = new OpenCvSharp.Rect(80, 950, 550 - 80, 990 - 950);
            // Rect r = new OpenCvSharp.Rect(120, 820, 450 - 120 , 850 - 820);
            OpenCvSharp.Mat box = new OpenCvSharp.Mat(new Size(boundingBox.Width, boundingBox.Height), MatType.CV_8U);
            //var horizontal_kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(25, 1));
            //g = g.MorphologyEx(MorphTypes.Open, horizontal_kernel, iterations: 2);
            //Cv2.ImShow("g", g);

            // Cv2.FindContours(g, out var cnts, out var hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);



            //Cv2.DrawContours(g, cnts, -1, new Scalar(255, 255, 255), 5, hierarchy: hierarchy);
            //Cv2.ImShow("gx", g);


            d2[boundingBox].CopyTo(box);

            Cv2.BitwiseNot(box, box);

            Mat labels = new Mat(), centroids = new Mat();
            Mat stats = new Mat();

            int cnt = Cv2.ConnectedComponentsWithStats(box, labels, stats, centroids, PixelConnectivity.Connectivity8);




            //   Cv2.FindContours(box, out var boxes, out var h2, RetrievalModes.List, ContourApproximationModes.ApproxNone);

            //   Cv2.CvtColor(box, box, ColorConversionCodes.GRAY2BGR);

            int areas = 0;

            for (var x = 1; x < stats.Size().Height; x++)
            {
                areas += stats.Get<int>(x, (int)ConnectedComponentsTypes.Area);
            }

            /*    for (var x = 0; x <= box.Size().Width; x++)
                {
                    for (var y = 0; y <= box.Size().Height; y++)
                    {




                        var l = labels.Get<int>(y, x);



                        if (l != 0)
                        {
                            var i = stats.Get<int>(l, (int)ConnectedComponentsTypes.Area);
                            areas += i;
                            //var z = box.Get<Vec3b>(y,x);
                            //z.Item0 = 0;
                            //z.Item1 = 0;
                            //z.Item2 = 255;
                            //box.Set(y, x, z);


                        }
                    }
                }
            */
            //Cv2.ImShow("aaa", box);
            //double maxCountour = 0.0;

            //foreach (var x in boxes)
            //{
            //    var r2 = Cv2.BoundingRect(x);
            //    if (r2.Width != box.Width && r2.Height != box.Height)
            //    {
            //        Cv2.Rectangle(box, r2, new Scalar(0, 255, 0), 2);
            //        var c = Cv2.ContourArea(x);
            //        if (c > 100)
            //            maxCountour += c;
            //    }
            //}

            //Point[] p = new Point[4];
            //p[0] = new Point(0, 0);
            //p[1] = new Point(box.Size().Width, 0);
            //p[2] = new Point(box.Size().Width, box.Size().Height);
            //p[3] = new Point(0, box.Size().Height);
            var boxarea = box.Size().Width * box.Size().Height;


            bool probableSiganture = false;
            if (areas > (boxarea / 50)) 
            {
                probableSiganture = true;
            }

            

            Cv2.ImShow(file  + " " + DateTimeOffset.Now.Ticks, box);
            return probableSiganture;
       
        }
    }
}
