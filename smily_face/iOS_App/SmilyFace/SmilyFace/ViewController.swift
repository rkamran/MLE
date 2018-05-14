//
//  ViewController.swift
//  SmilyFace
//
//  Created by Rashid Kamran on 5/13/18.
//  Copyright © 2018 Rashid Kamran. All rights reserved.
//

import UIKit
import AVKit
import CoreML
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    @IBOutlet var label: UILabel?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        setupCaptureSession()
        label?.text = "Sore Here"
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


    
    func setupCaptureSession() {
        // create a new capture session
        let captureSession = AVCaptureSession()
        
        // find the available cameras
        let availableDevices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video, position: .front).devices
        
        do {
            // select a camera
            for device in availableDevices {
                if device.position == .front {
                    captureSession.addInput(try AVCaptureDeviceInput(device: device))
                    break
                }
            }
        } catch {
            // print an error if the camera is not available
            print(error.localizedDescription)
        }
        
        // setup the video output to the screen and add output to our capture session
        let captureOutput = AVCaptureVideoDataOutput()
        captureSession.addOutput(captureOutput)
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.frame
        view.layer.addSublayer(previewLayer)
        
        // buffer the video and start the capture session
        captureOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.startRunning()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // load our CoreML Pokedex model
        guard let model = try? VNCoreMLModel(for: smilyFace().model) else { return }

        // run an inference with CoreML
        let request = VNCoreMLRequest(model: model) { (finishedRequest, error) in

            // grab the inference results
            guard let results = finishedRequest.results else { return }

            // grab the highest confidence result
            guard let Observation = results.first as? VNCoreMLFeatureValueObservation else { return }

            // create the label text components
//            let predclass = "\(Observation.identifier)"
//            let predconfidence = String(format: "%.02f%", Observation.confidence * 100)

            // set the label text
            DispatchQueue.main.async(execute: {
                //self.label.text = "\(predclass) \(predconfidence)"
                //let score = results.first
                //self.label?.text = score.description
                print(results.description)
                guard let score = (results as! [VNCoreMLFeatureValueObservation])[0].featureValue.multiArrayValue?[0].doubleValue else {return}
                
                if score < 2.5 {
                    self.label?.text = String("Low")
                } else if score > 2.5 && score < 4.0 {
                    self.label?.text = String("Good :)")
                } else if score >= 4.0 {
                    self.label?.text = String("Awesome :)")
                }
                
                
            })
        }

        // create a Core Video pixel buffer which is an image buffer that holds pixels in main memory
        // Applications generating frames, compressing or decompressing video, or using Core Image
        // can all make use of Core Video pixel buffers
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
    
        // execute the request
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
}

