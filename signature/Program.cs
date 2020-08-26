using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace signature
{
    class Program
    {
        static VectorGaussian wPosterior;
        static InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());

        static void SetupProbabilities()
        {
            double[] blockAreas = new double[] { 7500, 7500, 14500, 14500,  9900, 18800,    12960,  12960,  12960,  16000,  16000, 16000,   12270,  9900, 12000, 6875};
            double[] areas = new double[] {     1974, 1182,  851,   5535,   1000, 3112,     2348,   570,    40,     3259,   0,      32,     2176,   2326, 2137,  1679};
            bool[] signature = new bool[] {     true, true,  false, true,   true, true,     true,   false,  false,  true,   false,  false,  true,   true, true, true};

            Vector[] xdata = new Vector[areas.Length];
            for (int i = 0; i < xdata.Length; i++)
                xdata[i] = Vector.FromArray(areas[i], blockAreas[i], 1);
            VariableArray<Vector> x = Variable.Observed(xdata);

            // Create target y  
            VariableArray<bool> y = Variable.Observed(signature, x.Range);


            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3)));
            Range j = y.Range;
            double noise = 0.1;
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]), noise) > 0;




            



              wPosterior = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Dist over w=\n" + wPosterior);

            BinaryFormatter serializer = new BinaryFormatter();
            // write to disk  
            using (FileStream stream = new FileStream("temp.bin", FileMode.Create))
            {
                serializer.Serialize(stream, wPosterior);
            }
            // read from disk  
            using (FileStream stream = new FileStream("temp.bin", FileMode.Open))
            {
                wPosterior = (VectorGaussian)serializer.Deserialize(stream);

            }



         
           
        }


        public static void BayesPointMachine(double[] areas, double[] boxAreas, Variable<Vector> w, VariableArray<bool> y)
        { // Create x vector, augmented by 1 

            double noise = 0.1;
            Range j = y.Range; Vector[] xdata = new Vector[areas.Length];
            for (int i = 0; i < xdata.Length; i++)
                xdata[i] = Vector.FromArray(areas[i], boxAreas[i], 1);
            VariableArray<Vector> x = Variable.Observed(xdata, j); // Bayes Point Machine double noise = 0.1;  
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]), noise) > 0;
        }
        static void Main(string[] args)
        {

            SetupProbabilities();
              engine = new InferenceEngine(new ExpectationPropagation());
            string root = "";

           string name = "Seller1 Sign - Image background might cause issue.png";
           var b = RunOn(root + name, new OpenCvSharp.Rect(220, 230, 250, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(220, 270, 250, 30));
          
            Debug.Assert(b == true);  
            
            Console.WriteLine(name + "  seller   " + b);
              name = "Lien Relase 1 - Signature area contains seller1 sign contents.png";

              b = RunOn(root + name, new Rect(550, 800, 290, 50));
        
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lh1   " + b);


            b = RunOn(root + name, new Rect(50, 800, 290, 50));
            Console.WriteLine(name + "  seller   " + b);
            Debug.Assert(b == true);
            name = "Seller2 sign - Signature area contains seller1 sign contents.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(120, 820, 450 - 120, 850 - 820));
            Console.WriteLine(name + "  1 owner   " + b);
             
            Debug.Assert(b == true);
            b = RunOn(root + name, new OpenCvSharp.Rect(80, 950, 550 - 80, 990 - 950));
            Console.WriteLine(name + "  notary   " + b);
            Debug.Assert(b == true);

            name = "Lien Release 1 - Sign area 1 for IN format.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(60, 930, 360, 36));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  lr1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 930, 360, 36));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr2   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 670, 360, 36));
           // Cv2.WaitKey();
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr3   " + b);

            name = "Lien Release 1 - Sign area 2 for IN format.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(460, 600, 400, 40));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  lr1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(460, 760, 400, 40));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr2   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(460, 900, 400, 40));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  lr3   " + b);

            name = "Buyer Sign - Contains only scribling.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(480, 400, 409, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(560, 270, 330, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  seller   " + b);

            name = "Seller2 sign - Signature area contains seller1 sign contents and stamp_2.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(130, 820, 275, 25));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(75, 960, 480, 25));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);


            name = "Seller2 Sign - Signature area contains seller1 sign contents3.png";
            b = RunOn(root + name, new OpenCvSharp.Rect(20, 880, 420, 30));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(490, 950, 245, 15));
            Debug.Assert(b == false);
            Console.WriteLine(name + "  b1   " + b);
            b = RunOn(root + name, new OpenCvSharp.Rect(490, 695, 245, 15));
            Debug.Assert(b == true);
            Console.WriteLine(name + "  b1   " + b);
            Cv2.WaitKey();

        }
        static bool RunOn(string file, OpenCvSharp.Rect boundingBox)
        {
            //  OpenCvSharp.Mat m = new OpenCvSharp.Mat(root + "Seller2 sign - Signature area contains seller1 sign contents_2.png", OpenCvSharp.ImreadModes.Grayscale);
            OpenCvSharp.Mat m = new OpenCvSharp.Mat(file, OpenCvSharp.ImreadModes.Grayscale);
 
           // OpenCvSharp.Mat box2 = new OpenCvSharp.Mat(new Size(boundingBox.Width, boundingBox.Height), MatType.CV_8U);
            
          //  m[boundingBox].CopyTo(box2);
            
           // box2 = box2.AdaptiveThreshold(255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 3, 10);
          //  Cv2.ImShow("box2", box2);
            
            //var g = m.AdaptiveThreshold(255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv  , 3, 20);
            var g = m.Threshold(200, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            // g = g.Blur(new Size(5,5));

          //   Cv2.ImShow("ggg", g);

            var element = Cv2.GetStructuringElement(
                                MorphShapes.Rect,
                                new Size(10, 1));

            var h = g.MorphologyEx(MorphTypes.Close, element, iterations: 2);
            //    Cv2.ImShow("dilate", h);
            element = Cv2.GetStructuringElement(
                                MorphShapes.Rect,
                                new Size(80, 1));
            var mask = h.MorphologyEx(MorphTypes.Open, element, iterations: 2);
            //Cv2.ImShow("mask", mask);

            Mat newMask = new Mat();
            Cv2.BitwiseNot(mask, newMask);

            //  Cv2.ImShow("newMask", newMask);

            Mat newImage = new Mat();
            g.CopyTo(newImage, newMask);

      
            Cv2.BitwiseNot(newImage, newImage);
           // Cv2.ImShow("newImage", newImage);

            element = Cv2.GetStructuringElement(
                                 MorphShapes.Ellipse,
                                 new Size(2, 2));

            var d2 = newImage.MorphologyEx(MorphTypes.Dilate, element);
           //   Cv2.ImShow("d2", d2);
            element = Cv2.GetStructuringElement(
                             MorphShapes.Ellipse,
                             new Size(7, 7));
            d2 = d2.MorphologyEx(MorphTypes.Erode, element);
            // Cv2.ImShow("d3", d2);
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

            int[] quadrants = new int[4];
         
            int qh =  box.Size().Height / 2;
             int qw =  box.Size().Width / 2;

            var tl = new Rect(0, 0, qw, qh);
            var vl = new Rect(0, qh, qw, qh);
            var tr = new Rect(qw, 0, qw, qh);
            var br = new Rect(qw, qh, qw, qh);

            for (var x = 1; x < stats.Size().Height; x++)
            {
                var left = stats.Get<int>(x, (int)ConnectedComponentsTypes.Left);
                var top = stats.Get<int>(x, (int)ConnectedComponentsTypes.Top);
                var width = stats.Get<int>(x, (int)ConnectedComponentsTypes.Width);
                var height  = stats.Get<int>(x, (int)ConnectedComponentsTypes.Height);

                var re = new Rect(left, top, width, height);
                if(re.IntersectsWith(tl))
                {
                    quadrants[0] = 1;
                }
                if (re.IntersectsWith(vl))
                {
                    quadrants[1] = 1;
                }
                if (re.IntersectsWith(tr))
                {
                    quadrants[2] = 1;
                }
                if (re.IntersectsWith(br))
                {
                    quadrants[3] = 1;
                }

                //if (left < qw && top < qh)
                //{
                //    quadrants[0] = 1;
                //}
                //else if (left >= qw && top >= qh) 
                //{
                //    quadrants[3] = 1;
                //}
                //else if (left >= qw && top < qh)
                //{
                //    quadrants[2] = 1;
                //}
                //else if (left < qw && top >= qh)
                //{
                //    quadrants[1] = 1;
                //}

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


            double[] areasTest = new double[] { areas };
            double[] boxAreas = new double[] { boxarea };

            
            VariableArray<bool> ytest = Variable.Array<bool>(new Range(areasTest.Length));
            BayesPointMachine(areasTest, boxAreas, Variable.Random(wPosterior), ytest);
            var res = (DistributionStructArray<Bernoulli, bool>)engine.Infer(ytest);
            var mean = res[0].GetMean();


            Console.WriteLine(boxarea + " " + areas + " " + mean + " " + quadrants.Sum());
            bool probableSiganture = false;
            if (mean > 0.5 && quadrants.Sum() > 1)
            {
                probableSiganture = true;
            }



            Cv2.ImShow("box", box);
    
            return probableSiganture;

        }
    }
}
