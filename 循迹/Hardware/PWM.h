#ifndef __PWM_H
#define __PWM_H

#include "stm32f10x.h"                  // Device header

void PWM_Init();
void PWM_SetCompare_1(uint16_t Compare1);
void PWM_SetCompare_2(uint16_t Compare2);

#endif
