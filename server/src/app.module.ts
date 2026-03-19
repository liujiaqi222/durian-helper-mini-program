import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { CoreModule } from './core/core.module';
import { AiModule } from './modules/ai/ai.module';
import { DuriansModule } from './modules/durians/durians.module';

@Module({
  imports: [CoreModule, AiModule, DuriansModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
