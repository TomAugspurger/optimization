clear all
format long
kmax=5;
xk=[5;1]
Fxk=F(xk);
Bk=[10 2;1 1];
Ck=inv(Bk);
for k=0:kmax
    pk=-Ck*Fxk;
    xk=xk+pk
    Fxkp1=F(xk);
    yk=Fxkp1-Fxk;
    Ckyk=Ck*yk;
    pkTCk=pk'*Ck;
    Ck=Ck+(pk-Ckyk)*pkTCk/(pk'*Ckyk);    
    Fxk=Fxkp1;
   input('press return to continue')
end
