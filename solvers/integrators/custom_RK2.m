function xNext = custom_RK2(xNow, u, F, dt, P, M)

x(:,1) = xNow;
for i = 1:M
    k1 = dt * F(x(:,i),u) / 2.0;
    x(:,i+1) = x(:,i) + dt * F(x(:,i) + k1, u);
end
xNext = x(:,M+1);
    