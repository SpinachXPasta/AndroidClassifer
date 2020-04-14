package com.example.androidclassifer;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.content.Intent;
import android.net.Uri;
import android.util.Log;
import android.widget.TextView;
import android.os.Environment;

import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;

import org.opencv.android.Utils;
import org.opencv.core.CvType;


import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.provider.MediaStore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;



public class MainActivity extends AppCompatActivity {

    BaseLoaderCallback baseLoaderCallback;

    private static final int PICK_IMAGE_REQUEST = 1;
    private Button mButtonChooseImage;
    private Button mScoring;
    private EditText mEditTextFileName;
    private ImageView mImageView;
    private Uri mImageUri;
    private TextView mOutput;
    boolean startYolo = false;
    boolean firstTimeYolo = false;
    Net Model;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);




        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }

        };






        mButtonChooseImage = findViewById(R.id.button_choose_image);
        mScoring = findViewById(R.id.scoring);
        mEditTextFileName = findViewById(R.id.edit_text_file_name);
        mImageView = findViewById(R.id.image_view);
        mOutput = findViewById(R.id.Output);



        mOutput.setText("Score on Image");

        mButtonChooseImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openFileChooser();

            }
        });


        mScoring.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mImageUri != null) {
                    Scoring();
                }else{
                    mOutput.setText("Please upload Image first!");
                }

            }
        });

    }

    private void openFileChooser() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);


        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK
                && data != null && data.getData() != null) {
            mImageUri = data.getData();

            mImageView.setImageURI(mImageUri);

            //Try and load image from URI
            BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
            bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;







            Log.d("My storage dir","****** \n\n My Storage dir" + mImageUri.toString() + "\n\n\n ******");
        }
    }


    private void Scoring() {
        if (mImageUri != null){
            //always check if your app is given permission
            //String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/DNN_Config/yolov3-tiny-obj-2.cfg" ;
            //String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/DNN_Config/yolov3-tiny-obj_last-2.weights";
            //Model = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);


            try {
                Log.d("My storage dir","****** \n\n Init Bitmap");
                Bitmap bmp = MediaStore.Images.Media.getBitmap(this.getContentResolver(), mImageUri);
                Log.d("My storage dir","****** \n\n Executed Bitmap" + bmp.toString() + "\n\n\n ******");

                Mat obj = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC4);
                Utils.bitmapToMat(bmp, obj);

                Log.d("My storage dir","****** \n\n" + obj.toString());
                Log.d("My storage dir","****** \n\n Executed Mat");

                String Pbuf = Environment.getExternalStorageDirectory() + "/DNN_Config/frozen_mobile.pb";
                Model = Dnn.readNetFromTensorflow(Pbuf);
                //mOutput.setText("LOL");

                Imgproc.cvtColor(obj, obj, Imgproc.COLOR_RGBA2RGB);
                Mat imageBlob = Dnn.blobFromImage(obj, 0.00392, new Size(224,224),new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
                Model.setInput(imageBlob);

                java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

                List<String> outBlobNames = new java.util.ArrayList<>();
                outBlobNames.add(0, "avg_pool/Mean/flatten");
                outBlobNames.add(1, "dense_2/Softmax");

                Model.forward(result,outBlobNames);

                String holder = result.toString();
                Log.d("this is my Matrix", "ZZZ: " + holder + "\n\n\n ******succcesss!!!");

                Mat output_matrix = result.get(1);



                Core.MinMaxLocResult mm = Core.minMaxLoc(output_matrix);
                Point confidence = mm.maxLoc;
                String answer = confidence.toString();
                Log.d("this is my Matrix", "ZZZ: " + answer + "\n\n\n **2121****succcesss!!!");


                double[] finalX =  output_matrix.get(0,0);
                double[] finalY =  output_matrix.get(0,1);

                if (finalX[0] > finalY[0]){
                    mOutput.setText(
                            "Sport: BaseBall, Confidence: " + output_matrix.dump().split(",")[0].replace("[","").replace("]","")
                    );
                    Log.d("this is my Matrix", "ZZZ: " + output_matrix.dump() + "\n\n\n **21iosandsan*succcesss!!!");
                    Log.d("this is my Matrix", "ZZZ: " + finalX + "\n\n\n **asdsacccesss!!!");
                    Log.d("this is my Matrix", "\n\n\n  Final X \n\n\n");
                } else {
                    mOutput.setText(
                            "Sport: FootBall, Confidence: " + output_matrix.dump().split(",")[1].split(",")[0].replace("[","").replace("]","")
                    );
                    Log.d("this is my Matrix", "ZZZ: " + output_matrix.dump() + "\n\n\n **21iosandsan*succcesss!!!");
                    Log.d("this is my Matrix", "ZZZ: " + finalY + "\n\n\n **21iosawdqdqss!!!");
                    Log.d("this is my Matrix", "\n\n\n  Final Y \n\n\n");
                }



            }catch(Exception e){
                ;
            }





        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }



}
