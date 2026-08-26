## CODESYS in Automotive and Real-Time Control

**CODESYS** is a manufacturer-independent industrial automation and real-time control platform based on the **IEC 61131-3** programming standard. It supports Structured Text (ST), Function Block Diagram (FBD), Ladder Diagram (LD), and other IEC 61131-3 programming languages. CODESYS is used across industrial automation, motion control, robotics, mobile machines, and intelligent control applications.

Its relevance to automotive engineering comes from the increasing use of software-based control, real-time optimization, actuator control, automated manufacturing systems, and intelligent mobility technologies. Although automotive ECUs typically use specialized automotive development environments, CODESYS provides a practical platform for **control algorithm development, rapid prototyping, real-time controller implementation, test benches, and industrial/mobile-machine control**.


### About This Project

This repository demonstrates the implementation of **Grey Wolf Optimization (GWO)**, **Particle Swarm Optimization (PSO)** and optimization-based control algorithms in **CODESYS Structured Text** for automotive/vehicle-control applications. Along with this, a simulator with multiple controller for tuning the 4 wires of the vehicle: Steer by wire, Throttle by wire, Brake by wire, Gear by wire.

The objective is to demonstrate how computational intelligence algorithms can be implemented within an IEC 61131-3 real-time control environment and integrated with vehicle actuator models. The projects focus on:

* Grey Wolf Optimization (GWO)
* Particle Swarm Optimization (PSO)
* Automotive steering/control applications
* Real-time actuator control
* Optimization-based controller tuning
* CODESYS Structured Text (IEC 61131-3)
* Simulation and performance evaluation


This work provides a bridge between **academic optimization algorithms and practical real-time industrial control implementation**.
Particle Swarm Optimization based steering angle tracking system is built for adaptive Steer by wire systems inspired from : L. He et. al “An adaptive PI
controller by particle swarm optimization for angle tracking of steerby-wire,” IEEE/ASME Transactions on Mechatronics, vol. 27, no. 5, pp.3830–3840, 2022.
