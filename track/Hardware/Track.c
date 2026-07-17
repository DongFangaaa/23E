#include "stm32f10x.h"      
// Device header

float last_error = 0;

void Track_Init()
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
	
	GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOA, &GPIO_InitStructure);

  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15;
  GPIO_Init(GPIOB, &GPIO_InitStructure);
}

int16_t Track_State()
{
	 // 连续采样3次，三次一致才认为有效
    int16_t Sta1 = 0;
		int16_t Sta2 = 0;
		int16_t Sta3 = 0;
    
    // 第一次采样
    if(GPIO_ReadInputDataBit(GPIOA,GPIO_Pin_8))  Sta1 |= 0x10;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_12)) Sta1 |= 0x08;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_13)) Sta1 |= 0x04;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_14)) Sta1 |= 0x02;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_15)) Sta1 |= 0x01;
    
    // 第二次采样
    if(GPIO_ReadInputDataBit(GPIOA,GPIO_Pin_8))  Sta2 |= 0x10;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_12)) Sta2 |= 0x08;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_13)) Sta2 |= 0x04;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_14)) Sta2 |= 0x02;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_15)) Sta2 |= 0x01;
    
    // 第三次采样
    if(GPIO_ReadInputDataBit(GPIOA,GPIO_Pin_8))  Sta3 |= 0x10;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_12)) Sta3 |= 0x08;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_13)) Sta3 |= 0x04;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_14)) Sta3 |= 0x02;
    if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_15)) Sta3 |= 0x01;
    
    // 三次中至少两次一致才采用，否则返回上次结果
    static int16_t last_sta = 0x1F;
    if(Sta1 == Sta2 || Sta1 == Sta3)
        return last_sta = Sta1;
    if(Sta2 == Sta3)
        return last_sta = Sta2;
    return Sta1;  // 三次都不一致，返回上次稳定值
}

float Track_Error()
{
	uint8_t status = Track_State();
	switch(status)
	{
		case 0x1B:
			return last_error = 0;//11011
		case 0x13:
			return last_error = -1;//10011
		case 0x17:
			return last_error = -2;//10111
		case 0x07: 
			return last_error = -3;//00111
		case 0x05:
			return last_error = -3;//00101
		case 0x0F:
			return last_error = -4;//01111
		case 0x03:
			return last_error = -5;//00011
		//case 0x02:
			//return last_error = -5;//00010
		case 0x01:
			return last_error = -5;//00001
		case 0x19:
			return last_error = 1;//11001
		case 0x1D:
			return last_error = 2;//11101
		case 0x1c:
			return last_error = 3;//11100
		case 0x14:
			return last_error = 3;//10100
		case 0x1E:
			return last_error = 4;//11110
		case 0x18:
			return last_error = 5;//11000
		//case 0x08:
			//return last_error = 5;//01000
		case 0x10:
			return last_error = 5;//10000
		case 0x1F:
		{
			if(last_error >= -1.5f && last_error <= 1.5f)
      return last_error; // 不累加，直接保持
    
			// last_error较大，说明已经在持续跑偏，适度扩大
			if(last_error > 1.5f)  return (last_error + 0.5f > 5.0f) ? 5.0f : last_error + 0.5f;
			if(last_error < -1.5f) return (last_error - 0.5f < -5.0f) ? -5.0f : last_error - 0.5f;
			return 0;
		}//11111
		case 0x00:
			return last_error = 10;//00000
		default:
			return last_error;
	}
}
