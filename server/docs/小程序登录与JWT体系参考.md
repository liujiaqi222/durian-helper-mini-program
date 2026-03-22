# 小程序登录与 JWT 体系参考

本文档给 `durian-helper-mini-program/server` 参考，设计目标是把小程序登录链路拆清楚：

1. 小程序 `code` 用来向微信换 `openid`
2. `openid` 用来在服务端识别或创建用户
3. 服务端再签发自己的 JWT 作为登录态
4. 后续业务接口通过 JWT 识别当前用户，而不是让客户端反复上传 `openid`

这套设计参考当前 `ai-img-gen/server` 的实现思路，但文档按 `durian` 项目可直接落地的方式重写。

---

## 一、先讲结论

`openid` 和 JWT 不是一回事。

- `openid`：微信给你的小程序用户标识，只适合做“微信身份绑定”
- JWT：你自己服务端发的登录态凭证，只适合做“接口鉴权”

不要把 `openid` 当成全站 Bearer Token。原因很直接：

1. `openid` 不是你服务端签发的，无法证明“这次请求确实已经登录”
2. 如果客户端每次都传 `openid`，谁拿到这个值就可能伪装成该用户
3. 业务系统通常还需要内部 `userId` 和对外 `publicId`，不能把微信标识直接当业务主键到处传

推荐职责划分：

- `code`：一次性登录凭证，只在登录/刷新 session 时使用
- `openid`：微信侧身份标识，只存在服务端数据库中
- `session_key`：微信侧敏感会话材料，只在服务端保存，需要时刷新
- JWT：你自己的登录态票据，客户端后续请求统一走 `Authorization: Bearer <token>`
- `user`：服务端返回给前端的“当前用户公开资料”

---

## 二、推荐的数据模型

最少建议有一个 `users` 表，字段可以参考下面这套。

```ts
users {
  id: number;                 // 内部自增主键，只给服务端内部关联使用
  publicId: string;           // 对外用户标识，建议写入 JWT.sub
  openid: string | null;      // 小程序 openid，唯一
  unionid: string | null;     // 如果微信返回则存，唯一
  sessionKey: string | null;  // code2session 换回来的 session_key
  name: string | null;        // 用户昵称或自定义资料
  phone: string | null;       // 手机号
  createdAt: Date;
  updatedAt: Date;
}
```

建议约束：

- `id` 是内部主键，不对 C 端暴露
- `publicId` 唯一，对外展示和 JWT `sub` 都用它
- `openid` 唯一
- `unionid` 唯一，但允许为空
- `sessionKey` 允许为空，但只保存在服务端

推荐做“内外 ID 分离”：

- 内部关系和数据库关联继续用 `id`
- 前端和日志尽量只看 `publicId`
- JWT 里不要直接塞数据库自增 `id`

---

## 三、完整登录链路

### 1. 小程序端拿 `code`

前端调用：

```ts
wx.login({
  success(res) {
    const code = res.code;
  }
});
```

这个 `code` 是一次性的，应该马上发给服务端。

### 2. 服务端调用微信 `code2session`

服务端请求：

```http
GET https://api.weixin.qq.com/sns/jscode2session
  ?appid=APPID
  &secret=SECRET
  &js_code=CODE
  &grant_type=authorization_code
```

微信典型返回：

```json
{
  "openid": "OPENID",
  "session_key": "SESSION_KEY",
  "unionid": "UNIONID"
}
```

如果失败则可能返回：

```json
{
  "errcode": 40029,
  "errmsg": "invalid code"
}
```

### 3. 用 `openid/unionid` 查找用户

逻辑建议：

1. 先按 `openid` 查用户
2. 如果有 `unionid`，也可以同时支持按 `unionid` 命中
3. 查不到则创建用户
4. 查到了则更新最新的 `sessionKey`

注意：

- `openid` 是识别“这是哪个微信用户”的关键字段
- 但登录完成后，不要让客户端继续靠 `openid` 调业务接口

### 4. 为用户补齐业务身份

创建新用户时，建议同时生成：

- `publicId`
- 其他你们业务要展示的公开字段

这样登录返回的 `user` 就不需要暴露内部数据库主键。

### 5. 服务端签发 JWT

推荐 payload：

```ts
type UserJwtPayload = {
  sub: string; // 存 publicId
};
```

签发示例：

```ts
const payload = { sub: user.publicId };
const token = jwtService.sign(payload, { secret: JWT_SECRET });
```

### 6. 登录接口返回 `token + user`

推荐返回结构：

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "publicId": "6024378002",
    "name": null,
    "phone": null,
    "createdAt": "2026-03-22T00:00:00.000Z",
    "updatedAt": "2026-03-22T00:00:00.000Z"
  }
}
```

这里的含义要清楚：

- `token`：给前端保存，后面请求带上
- `user`：给前端展示当前用户信息

---

## 四、推荐接口设计

### 1. 登录

`POST /auth/login`

请求：

```json
{
  "code": "wx-login-code"
}
```

返回：

```json
{
  "token": "jwt-token",
  "user": {
    "publicId": "1234567890",
    "name": null,
    "phone": null
  }
}
```

### 2. 当前用户信息

`GET /users/me`

请求头：

```http
Authorization: Bearer <jwt>
```

返回：

```json
{
  "publicId": "1234567890",
  "name": "张三",
  "phone": null
}
```

### 3. 刷新微信 `session_key`（可选）

如果后面你们也要做微信支付、手机号解密、敏感接口，建议保留：

`POST /auth/refresh-session`

请求头：

```http
Authorization: Bearer <jwt>
```

请求体：

```json
{
  "code": "new-wx-login-code"
}
```

返回：

```json
{
  "success": true
}
```

用途：

- 小程序端再次 `wx.login()` 获取新 `code`
- 服务端重新换取 `session_key`
- 用于支付前、解密手机号前等需要新鲜 `session_key` 的场景

---

## 五、JWT 在服务端到底怎么用

推荐做法是：

1. 前端每次调用受保护接口都带上 `Authorization: Bearer <jwt>`
2. 服务端在 Guard / Strategy 里校验 token
3. 从 JWT 的 `sub` 查出当前用户
4. 把当前用户上下文挂到 `req.user`
5. Controller 和 Service 继续用内部 `userId`

推荐校验结果结构：

```ts
type AuthenticatedUser = {
  userId: number;   // 内部主键
  publicId: string; // 对外标识
};
```

推荐策略逻辑：

```ts
async validate(payload: { sub: string }) {
  if (!payload?.sub) {
    throw new UnauthorizedException('invalid token');
  }

  const user = await findUserByPublicId(payload.sub);
  if (!user) {
    throw new UnauthorizedException('user not found');
  }

  return {
    userId: user.id,
    publicId: user.publicId,
  };
}
```

这样业务代码会比较干净：

```ts
@UseGuards(JwtAuthGuard)
@Get('me')
getMe(@Request() req) {
  return this.usersService.getProfile(req.user.userId);
}
```

这里最关键的一点是：

- JWT 负责把“请求是谁”解析成 `req.user`
- `openid` 不直接进入大部分业务接口

---

## 六、为什么推荐 `sub = publicId`，而不是 `sub = openid`

如果 JWT 直接写 `openid`，短期也能用，但不够稳妥。

推荐写 `publicId` 的原因：

1. `openid` 是微信生态字段，不应该变成业务系统的统一外显主键
2. 业务以后可能接手机号、游客态、Web 登录、后台导入用户，不一定都天然有 `openid`
3. `publicId` 更适合对外展示、埋点、日志、客服排查
4. 内部 `id`、外部 `publicId`、第三方 `openid` 分层更清晰

推荐映射关系：

- 微信身份层：`openid` / `unionid`
- 业务公开身份层：`publicId`
- 数据库内部关系层：`id`

---

## 七、`openid`、JWT、`user` 三者的边界

### `openid`

作用：

- 唯一标识微信小程序用户
- 绑定微信支付、微信解密、微信能力调用
- 只建议保存在服务端数据库

不建议：

- 当通用业务接口的鉴权凭证
- 在前端四处透传

### JWT

作用：

- 作为你自己服务端的登录态票据
- 每次请求时证明“当前用户已登录”
- 让服务端快速恢复当前用户上下文

不建议：

- 放太多业务字段
- 直接存敏感资料

### `user`

作用：

- 返回给前端做页面展示和当前用户状态同步
- 是“当前用户公开视图”

推荐只返回业务需要的公开字段，例如：

- `publicId`
- `name`
- `phone`
- `avatar`
- `createdAt`

不要把下面这些直接返回给前端：

- `sessionKey`
- 数据库内部 `id`
- 过多内部状态字段

---

## 八、适合 `durian` 项目的最小实现

如果 `durian` 现在只做小程序登录，不做支付，可以先上最小版。

### 第一步：表结构

至少保留：

- `id`
- `publicId`
- `openid`
- `unionid`
- `sessionKey`
- `name`
- `createdAt`
- `updatedAt`

### 第二步：登录接口

实现 `POST /auth/login`：

1. 接收前端 `code`
2. 调微信 `code2session`
3. 用 `openid/unionid` 查或建用户
4. 生成 JWT
5. 返回 `token + user`

### 第三步：用户鉴权

实现：

- `JwtStrategy`
- `JwtAuthGuard`
- `GET /users/me`

### 第四步：前端接入

前端流程：

1. `wx.login()` 拿 `code`
2. 调 `/auth/login`
3. 保存后端返回的 `token`
4. 后续请求统一带 `Authorization: Bearer <token>`
5. 启动时可调 `/users/me` 恢复当前登录用户

---

## 九、NestJS 参考伪代码

### DTO

```ts
export class LoginDto {
  code: string;
}
```

### 登录 service

```ts
async login(dto: LoginDto) {
  const wx = await this.wechatService.code2Session(dto.code);

  let user = await this.userRepo.findByOpenidOrUnionid(wx.openid, wx.unionid);

  if (!user) {
    user = await this.userRepo.create({
      openid: wx.openid,
      unionid: wx.unionid,
      sessionKey: wx.session_key,
      publicId: generatePublicId(),
    });
  } else {
    user = await this.userRepo.updateSessionKey(user.id, wx.session_key);
  }

  const token = this.jwtService.sign({ sub: user.publicId });

  return {
    token,
    user: {
      publicId: user.publicId,
      name: user.name,
    },
  };
}
```

### JWT strategy

```ts
async validate(payload: { sub: string }) {
  const user = await this.userRepo.findByPublicId(payload.sub);
  if (!user) throw new UnauthorizedException();

  return {
    userId: user.id,
    publicId: user.publicId,
  };
}
```

### 受保护接口

```ts
@UseGuards(JwtAuthGuard)
@Get('me')
getMe(@Request() req) {
  return this.usersService.getProfile(req.user.userId);
}
```

---

## 十、实现时容易踩的坑

### 1. 把 `openid` 当 token 用

这是最常见的误区。`openid` 只能说明“这是微信里的谁”，不能说明“这次请求已经被你服务端认证过”。

### 2. JWT 里直接放数据库自增 `id`

可以用，但不够理想。更推荐放 `publicId`，把内部主键和外部身份隔离开。

### 3. 把 `session_key` 返回给前端

不建议。`session_key` 应该只存在服务端。

### 4. 不保留 `session_key`

如果后面要做微信支付、手机号解密、某些微信能力调用，你会需要它。现在先存上，后续扩展更顺。

### 5. 登录后不提供 `/users/me`

很多前端在冷启动时需要恢复当前用户信息。没有 `/users/me`，前端通常会把登录态和用户资料耦死在本地缓存里，后面会变乱。

---

## 十一、推荐的最终落地形态

对 `durian`，建议最终采用下面这套口径：

1. 小程序前端只负责拿 `code`
2. 服务端负责拿 `openid`
3. 服务端负责查找或创建 `user`
4. 服务端负责签发 JWT
5. 前端以后只拿 JWT 调业务接口
6. 支付、手机号解密等微信能力由服务端用库里的 `openid/sessionKey` 完成

对应一句话总结：

> `code` 用来换微信身份，`openid` 用来绑定微信用户，JWT 用来承载你自己系统的登录态，`user` 用来给前端展示当前用户资料。

