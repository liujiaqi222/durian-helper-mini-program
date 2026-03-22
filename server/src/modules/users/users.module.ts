import { Module, forwardRef } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AuthModule } from '../auth/auth.module';
import { USERS_REPOSITORY } from './users.constants';
import { InMemoryUsersRepository } from './users.memory-repository';
import { DrizzleUsersRepository } from './users.repository';
import { UsersController } from './users.controller';
import { UsersService } from './users.service';

@Module({
  imports: [forwardRef(() => AuthModule)],
  controllers: [UsersController],
  providers: [
    UsersService,
    DrizzleUsersRepository,
    InMemoryUsersRepository,
    {
      provide: USERS_REPOSITORY,
      inject: [ConfigService, DrizzleUsersRepository, InMemoryUsersRepository],
      useFactory: (
        configService: ConfigService,
        drizzleRepository: DrizzleUsersRepository,
        memoryRepository: InMemoryUsersRepository,
      ) => {
        return configService.get<string>('environment') === 'test'
          ? memoryRepository
          : drizzleRepository;
      },
    },
  ],
  exports: [UsersService, USERS_REPOSITORY],
})
export class UsersModule {}
