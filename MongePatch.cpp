#include <iostream>
#include "MongePatch.h"
#include <Eigen/Dense>
#include <Eigen/LU>

// Given a point P, its normal, and its closest neighbors (including itself) 
// compute a quadratic Monge patch that approximates the neighborhood of P.
// The resulting patch will be used to compute the principal curvatures of the 
// surface at point P.

void MongePatch::init(const glm::vec3 &P, const glm::vec3 &normal, const vector<glm::vec3> &closest)
{
	 
	Eigen::MatrixXf _matrix = Eigen::MatrixXf(6,6);
	Eigen::VectorXf b(6);
	b.setZero();
	Eigen::VectorXf s(6);

	Eigen::Matrix2f final;
	
////1crate new coo system 
	glm::vec3 W =- normal;
	glm::vec3 V = glm::vec3(0, -W.y, W.z);
	glm::vec3 U = glm::cross(W, V);
	

////2transfer all KNN points to this system
	for (size_t i = 0; i < closest.size(); i++)
	{
		glm::vec3  diffrence =glm::vec3(closest[i].x - P.x, closest[i].y - P.y, closest[i].z - P.z);
		
		float Xcomponent = glm::dot( diffrence,V);
		float Ycomponent = glm::dot(diffrence,U);
		float Zcomponent = glm::dot(diffrence,W);
		
		Eigen::VectorXf temp(6);
		temp.setZero();

		temp(0) = Xcomponent * Xcomponent ;
		temp(1) = Xcomponent * Ycomponent;
		temp(2) = Ycomponent * Ycomponent;
		temp(3) = Xcomponent;
		temp(4) = Ycomponent;
		temp(5)= 1;

		_matrix += temp * temp.transpose();
		b(0) += temp(0) * Zcomponent;
		b(1) += temp(1) * Zcomponent;
		b(2) += temp(2) * Zcomponent;
		b(3) += temp(3) * Zcomponent;
		b(4) += temp(4) * Zcomponent;
		b(5) += temp(5) * Zcomponent;

	}
      s=_matrix.colPivHouseholderQr().solve(b);

	  final << 2 * s(0), s(1), s(1), 2 * s(2);
	  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigensolver(final);
	   min = eigensolver.eigenvalues()[0];
	   max = eigensolver.eigenvalues()[1];

}

// Return the values of the two principal curvatures for this patch

void MongePatch::principalCurvatures(float &kmin, float &kmax) const
{
	kmin =min;
	kmax = max;
}


