/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  num_particles = 10;
  
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {
    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen);
    
    Particle p;
    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1;
    
    particles.push_back(p);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  double v_yaw = velocity/yaw_rate;
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    double x = p.x;
    double y = p.y;
    double theta = p.theta;
    
    double new_x = x + v_yaw * (sin(theta + yaw_rate * delta_t) - sin(theta));
    double new_y = y + v_yaw * (cos(theta) - cos(theta + yaw_rate * delta_t));
    double new_theta = theta + yaw_rate * delta_t;
    
    normal_distribution<double> dist_x(new_x, std_x);
    normal_distribution<double> dist_y(new_y, std_y);
    normal_distribution<double> dist_theta(new_theta, std_theta);
    
    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen);
   
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    
    particles[i] = p;
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs obs = observations[i];
    double x1 = obs.x;
    double y1 = obs.y;
    double min = numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      double x2 = pred.x;
      double y2 = pred.y;
      
      double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
      if (distance < min){
        obs.id = pred.id;
      }
    }
    observations[i] = obs;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  
  /*
   * 1. Take vehicle observations and translate each into map coordinates
   * 2. Look for the nearest landmark to the transformed observation
   * 3. Calculate the probability using the multi-variate gaussian dist
   * 4. Repeat 1-3 for each observation
   * 5. Multiply each of the probabilities together - this is the new weight
   */
  
  for (Particle& p : particles) {
    
    double weight = 1;
    
    for (LandmarkObs obs : observations) {
      // Translate vehicle observation coordinates into map coordinates  
      double map_x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
      double map_y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
      
      // look for nearest landmark to the transformed observation
      double min_dist = numeric_limits<double>::max();
      Map::single_landmark_s nearest_landmark;
      for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
        double distance = dist(landmark.x_f, landmark.y_f, map_x, map_y);
        if (distance < min_dist) {
          min_dist = distance;
          nearest_landmark = landmark;
        }
      }
      
      //Compute multi-variate Gaussian distribution
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double dividend = 2. * M_PI * sig_x * sig_y;
      double x_term = pow(obs.x - nearest_landmark.x_f, 2) / (2. * pow(sig_x, 2));
      double y_term = pow(obs.y - nearest_landmark.y_f, 2) / (2. * pow(sig_y, 2));
      
      double prob = 1/dividend * exp(-(x_term + y_term));
      weight *= prob;
      
    }
    
    p.weight = weight;
  }
  
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
