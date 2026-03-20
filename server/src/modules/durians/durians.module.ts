import { Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AiModule } from '../ai/ai.module';
import { UploadsModule } from '../uploads/uploads.module';
import { DURIAN_ANALYSIS_REPOSITORY } from './durians.constants';
import { CvService } from './cv.service';
import { DuriansController } from './durians.controller';
import { InMemoryDurianAnalysisRepository } from './durians.memory-repository';
import { DrizzleDurianAnalysisRepository } from './durians.repository';
import { DuriansService } from './durians.service';

@Module({
  imports: [AiModule, UploadsModule],
  controllers: [DuriansController],
  providers: [
    DuriansService,
    CvService,
    DrizzleDurianAnalysisRepository,
    InMemoryDurianAnalysisRepository,
    {
      provide: DURIAN_ANALYSIS_REPOSITORY,
      inject: [
        ConfigService,
        DrizzleDurianAnalysisRepository,
        InMemoryDurianAnalysisRepository,
      ],
      useFactory: (
        configService: ConfigService,
        drizzleRepository: DrizzleDurianAnalysisRepository,
        memoryRepository: InMemoryDurianAnalysisRepository,
      ) => {
        return configService.get<string>('environment') === 'test'
          ? memoryRepository
          : drizzleRepository;
      },
    },
  ],
  exports: [DuriansService],
})
export class DuriansModule {}
